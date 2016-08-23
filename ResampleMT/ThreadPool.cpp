// ThreadPoolDLL.cpp : définit les fonctions exportées pour l'application DLL.
//

#include "ThreadPool.h"

#define VERSION "ThreadPool 1.0.1"


#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseHandle(ptr); ptr=NULL;}

//#define MAX_USERS 500



// Helper function to count set bits in the processor mask.
static uint8_t CountSetBits(ULONG_PTR bitMask)
{
    DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
    uint8_t bitSetCount = 0;
    ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT;    
    DWORD i;
    
    for (i = 0; i <= LSHIFT; ++i)
    {
        bitSetCount += ((bitMask & bitTest)?1:0);
        bitTest/=2;
    }

    return bitSetCount;
}


static void Get_CPU_Info(Arch_CPU& cpu)
{
    bool done = false;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer=NULL;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr=NULL;
    DWORD returnLength=0;
    uint8_t logicalProcessorCount=0;
    uint8_t processorCoreCount=0;
    DWORD byteOffset=0;

	cpu.NbLogicCPU=0;
	cpu.NbPhysCore=0;

    while (!done)
    {
        BOOL rc=GetLogicalProcessorInformation(buffer, &returnLength);

        if (rc==FALSE) 
        {
            if (GetLastError()==ERROR_INSUFFICIENT_BUFFER) 
            {
                myfree(buffer);
                buffer=(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength);

                if (buffer==NULL) return;
            } 
            else
			{
				myfree(buffer);
				return;
			}
        } 
        else done=true;
    }

    ptr=buffer;

    while ((byteOffset+sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION))<=returnLength) 
    {
        switch (ptr->Relationship) 
        {
			case RelationProcessorCore :
	            // A hyperthreaded core supplies more than one logical processor.
				cpu.NbHT[processorCoreCount]=CountSetBits(ptr->ProcessorMask);
		        logicalProcessorCount+=cpu.NbHT[processorCoreCount];
				cpu.ProcMask[processorCoreCount++]=ptr->ProcessorMask;
			    break;
			default : break;
        }
        byteOffset+=sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }
	free(buffer);

	cpu.NbPhysCore=processorCoreCount;
	cpu.NbLogicCPU=logicalProcessorCount;
}


static ULONG_PTR GetCPUMask(ULONG_PTR bitMask, uint8_t CPU_Nb)
{
    uint8_t LSHIFT=sizeof(ULONG_PTR)*8-1;
    uint8_t i=0,bitSetCount=0;
    ULONG_PTR bitTest=1;    

	CPU_Nb++;
	while (i<=LSHIFT)
	{
		if ((bitMask & bitTest)!=0) bitSetCount++;
		if (bitSetCount==CPU_Nb) return(bitTest);
		else
		{
			i++;
			bitTest<<=1;
		}
	}
	return(0);
}


static void CreateThreadsMasks(Arch_CPU cpu, ULONG_PTR *TabMask,uint8_t NbThread,uint8_t offset)
{
	memset(TabMask,0,NbThread*sizeof(ULONG_PTR));

	if ((cpu.NbLogicCPU==0) || (cpu.NbPhysCore==0)) return;

	uint8_t current_thread=0;

	for(uint8_t i=0; i<cpu.NbPhysCore; i++)
	{
		uint8_t Nb_Core_Th=NbThread/cpu.NbPhysCore+( ((NbThread%cpu.NbPhysCore)>i) ? 1:0 );

		if (Nb_Core_Th>0)
		{
			const uint8_t offs=(cpu.NbHT[i]>offset) ? offset:cpu.NbHT[i]-1;

			for(uint8_t j=0; j<Nb_Core_Th; j++)
				TabMask[current_thread++]=GetCPUMask(cpu.ProcMask[i],(j+offs)%cpu.NbHT[i]);
		}
	}
}


DWORD WINAPI ThreadPool::StaticThreadpool(LPVOID lpParam )
{
	MT_Data_Thread *data=(MT_Data_Thread *)lpParam;
	
	while (true)
	{
		WaitForSingleObject(data->nextJob,INFINITE);
		switch(data->f_process)
		{
			case 1 :
				if (data->MTData!=NULL)
				{
					data->MTData->thread_Id=data->thread_Id;
					data->MTData->pFunc(data->MTData);
				}
				break;
			case 255 : return(0); break;
			default : break;
		}
		ResetEvent(data->nextJob);
		SetEvent(data->jobFinished);
	}
}


ThreadPool& ThreadPool::Init(uint8_t num)
{
	static ThreadPool Pool[MAX_THREAD_POOL];

	if (num>=MAX_THREAD_POOL) num=0;
	return(Pool[num]);
}


ThreadPool::ThreadPool(void):Status_Ok(true)
{
	int16_t i;

	for (i=0; i<MAX_MT_THREADS; i++)
	{
		MT_Thread[i].MTData=NULL;
		MT_Thread[i].f_process=0;
		MT_Thread[i].thread_Id=(uint8_t)i;
		MT_Thread[i].jobFinished=NULL;
		MT_Thread[i].nextJob=NULL;
		thds[i]=NULL;
	}
//	memset(TabId,0,MAX_USERS*sizeof(DWORD));
	CSectionOk=FALSE;
	JobsEnded=NULL;
	ThreadPoolFree=NULL;
	TotalThreadsRequested=0;
	CurrentThreadsAllocated=0;
	CurrentThreadsUsed=0;
	NbreUsers=0;
	JobsRunning=false;
	ThreadPoolRequested=false;

	Get_CPU_Info(CPU);
	if ((CPU.NbLogicCPU==0) || (CPU.NbPhysCore==0))
	{
		Status_Ok=false;
		return;
	}

	CSectionOk=InitializeCriticalSectionAndSpinCount(&CriticalSection,0x00000040);
	if (CSectionOk==FALSE)
	{
		Status_Ok=false;
		return;
	}

	JobsEnded=CreateEvent(NULL,TRUE,TRUE,NULL);
	if (JobsEnded==NULL)
	{
		FreeData();
		Status_Ok=false;
		return;
	}

	ThreadPoolFree=CreateEvent(NULL,TRUE,TRUE,NULL);
	if (ThreadPoolFree==NULL)
	{
		FreeData();
		Status_Ok=false;
	}
}


ThreadPool::~ThreadPool(void)
{
	Status_Ok=false;
	FreeData();
}


void ThreadPool::FreeThreadPool(void) 
{
	int16_t i;

	for (i=TotalThreadsRequested-1; i>=0; i--)
	{
		if (thds[i]!=NULL)
		{
			MT_Thread[i].f_process=255;
			SetEvent(MT_Thread[i].nextJob);
			WaitForSingleObject(thds[i],INFINITE);
			myCloseHandle(thds[i]);
		}
	}

	for (i=TotalThreadsRequested-1; i>=0; i--)
	{
		myCloseHandle(MT_Thread[i].nextJob);
		myCloseHandle(MT_Thread[i].jobFinished);
	}

	TotalThreadsRequested=0;
	CurrentThreadsAllocated=0;
	CurrentThreadsUsed=0;
	JobsRunning=false;
	ThreadPoolRequested=false;
}


void ThreadPool::FreeData(void) 
{
	myCloseHandle(ThreadPoolFree);
	myCloseHandle(JobsEnded);
	if (CSectionOk==TRUE)
	{
		DeleteCriticalSection(&CriticalSection);
		CSectionOk=FALSE;
	}
}


uint8_t ThreadPool::GetThreadNumber(uint8_t thread_number,bool logical)
{
	const uint8_t nCPU=(logical) ? CPU.NbLogicCPU:CPU.NbPhysCore;

	if (thread_number==0) return((nCPU>MAX_MT_THREADS) ? MAX_MT_THREADS:nCPU);
	else return(thread_number);
}


bool ThreadPool::AllocateThreads(DWORD pId,uint8_t thread_number,uint8_t offset)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if (thread_number==0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

/*	if (NbreUsers>=MAX_USERS)
	{
		ReleaseMutex(ghMutex);
		return(false);
	}

	uint16_t i=0;
	while ((NbreUsers>i) && (TabId[i]!=pId)) i++;
	if (i==NbreUsers)
	{
		TabId[i]=pId;
		NbreUsers++;
	}*/
	NbreUsers++;

	if (thread_number>CurrentThreadsAllocated)
	{
		TotalThreadsRequested=thread_number;
		while (JobsRunning)
		{
			LeaveCriticalSection(&CriticalSection);
			WaitForSingleObject(JobsEnded,INFINITE);
			EnterCriticalSection(&CriticalSection);
		}
		CreateThreadPool(offset);
		if (!Status_Ok) return(false);
	}

	LeaveCriticalSection(&CriticalSection);

	return(true);
}


bool ThreadPool::DeAllocateThreads(DWORD pId)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if (NbreUsers==0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(true);
	}

/*	uint16_t i=0;
	while ((NbreUsers>i) && (TabId[i]!=pId)) i++;
	if (i==NbreUsers)
	{
		ReleaseMutex(ghMutex);
		return(false);
	}

	if (i<NbreUsers-1)
	{
		for(uint16_t j=i+1; j<NbreUsers; j++)
			TabId[j-1]=TabId[j];
	}
	NbreUsers--;
	TabId[NbreUsers]=0;*/
	NbreUsers--;

	if (NbreUsers==0) FreeThreadPool();

	LeaveCriticalSection(&CriticalSection);

	return(true);
}



void ThreadPool::CreateThreadPool(uint8_t offset)
{
	int16_t i;

	if (CurrentThreadsAllocated>0)
	{
		for(i=0; i<CurrentThreadsAllocated; i++)
			SuspendThread(thds[i]);
	}

	CreateThreadsMasks(CPU,ThreadMask,TotalThreadsRequested,offset);

	for(i=0; i<CurrentThreadsAllocated; i++)
	{
		SetThreadAffinityMask(thds[i],ThreadMask[i]);
		ResumeThread(thds[i]);
	}

	i=CurrentThreadsAllocated;
	while ((i<TotalThreadsRequested) && Status_Ok)
	{
		MT_Thread[i].jobFinished=CreateEvent(NULL,TRUE,TRUE,NULL);
		MT_Thread[i].nextJob=CreateEvent(NULL,TRUE,FALSE,NULL);
		Status_Ok=Status_Ok && ((MT_Thread[i].jobFinished!=NULL) && (MT_Thread[i].nextJob!=NULL));
		i++;
	}
	if (!Status_Ok)
	{
		FreeThreadPool();
		LeaveCriticalSection(&CriticalSection);
		FreeData();
		return;
	}

	i=CurrentThreadsAllocated;
	while ((i<TotalThreadsRequested) && Status_Ok)
	{
		thds[i]=CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)StaticThreadpool,&MT_Thread[i],CREATE_SUSPENDED,&tids[i]);
		Status_Ok=Status_Ok && (thds[i]!=NULL);
		if (Status_Ok)
		{
			SetThreadAffinityMask(thds[i],ThreadMask[i]);
			ResumeThread(thds[i]);
		}
		i++;
	}
	if (!Status_Ok)
	{
		FreeThreadPool();
		LeaveCriticalSection(&CriticalSection);
		FreeData();
		return;
	}

	CurrentThreadsAllocated=TotalThreadsRequested;
}


bool ThreadPool::RequestThreadPool(DWORD pId,uint8_t thread_number,Public_MT_Data_Thread *Data)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if (thread_number>CurrentThreadsAllocated)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	while (ThreadPoolRequested)
	{
		LeaveCriticalSection(&CriticalSection);
		WaitForSingleObject(ThreadPoolFree,INFINITE);
		EnterCriticalSection(&CriticalSection);
	}

	for(uint8_t i=0; i<thread_number; i++)
		MT_Thread[i].MTData=Data+i;

	CurrentThreadsUsed=thread_number;

	ThreadPoolRequestProcessId=pId;

	ThreadPoolRequested=true;
	ResetEvent(ThreadPoolFree);

	LeaveCriticalSection(&CriticalSection);

	return(true);	
}


bool ThreadPool::ReleaseThreadPool(DWORD pId)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if (ThreadPoolRequestProcessId!=pId)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (ThreadPoolRequested)
	{
		for(uint8_t i=0; i<CurrentThreadsUsed; i++)
			MT_Thread[i].MTData=NULL;
		CurrentThreadsUsed=0;
		ThreadPoolRequested=false;
		ThreadPoolRequestProcessId=0;
		SetEvent(ThreadPoolFree);
	}

	LeaveCriticalSection(&CriticalSection);

	return(true);
}


bool ThreadPool::StartThreads(DWORD pId)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if ((!ThreadPoolRequested) || (CurrentThreadsUsed==0) || (ThreadPoolRequestProcessId!=pId))
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (JobsRunning)
	{
		LeaveCriticalSection(&CriticalSection);
		return(true);
	}

	JobsRunning=true;
	ResetEvent(JobsEnded);

	for(uint8_t i=0; i<CurrentThreadsUsed; i++)
	{
		MT_Thread[i].f_process=1;
		ResetEvent(MT_Thread[i].jobFinished);
		SetEvent(MT_Thread[i].nextJob);
	}

	LeaveCriticalSection(&CriticalSection);
	return(true);	
}


bool ThreadPool::WaitThreadsEnd(DWORD pId)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if ((!ThreadPoolRequested) || (CurrentThreadsUsed==0) || (ThreadPoolRequestProcessId!=pId))
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (!JobsRunning)
	{
		LeaveCriticalSection(&CriticalSection);
		return(true);
	}

	for(uint8_t i=0; i<CurrentThreadsUsed; i++)
		WaitForSingleObject(MT_Thread[i].jobFinished,INFINITE);

	for(uint8_t i=0; i<CurrentThreadsUsed; i++)
		MT_Thread[i].f_process=0;

	JobsRunning=false;
	SetEvent(JobsEnded);

	LeaveCriticalSection(&CriticalSection);

	return(true);
}

