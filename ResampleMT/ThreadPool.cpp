// ThreadPoolDLL.cpp : définit les fonctions exportées pour l'application DLL.
//

#include "ThreadPool.h"

#define VERSION "ThreadPool 1.1.0"


#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseHandle(ptr); ptr=NULL;}


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




ThreadPool::ThreadPool(HANDLE _JobsEnded, HANDLE _ThreadPoolFree,bool *_ThreadPoolRequested, bool *_JobsRunning):
Status_Ok(true),JobsEnded(_JobsEnded),ThreadPoolFree(_ThreadPoolFree),
JobsRunning(_JobsRunning),ThreadPoolRequested(_ThreadPoolRequested)
{
	int16_t i;

	for (i=0; i<MAX_MT_THREADS; i++)
	{
		jobFinished[i]=NULL;
		nextJob[i]=NULL;
		MT_Thread[i].MTData=NULL;
		MT_Thread[i].f_process=0;
		MT_Thread[i].thread_Id=(uint8_t)i;
		MT_Thread[i].jobFinished=NULL;
		MT_Thread[i].nextJob=NULL;
		thds[i]=NULL;
	}
	TotalThreadsRequested=0;
	CurrentThreadsAllocated=0;
	CurrentThreadsUsed=0;
	*JobsRunning=false;
	*ThreadPoolRequested=false;

	Get_CPU_Info(CPU);
	if ((CPU.NbLogicCPU==0) || (CPU.NbPhysCore==0))
	{
		Status_Ok=false;
		return;
	}

	if ((JobsEnded==NULL) || (ThreadPoolFree==NULL))
	{
		Status_Ok=false;
		return;
	}
}



void ThreadPool::FreeThreadPool(void) 
{
	int16_t i;

	for (i=TotalThreadsRequested-1; i>=0; i--)
	{
		if (thds[i]!=NULL)
		{
			MT_Thread[i].f_process=255;
			SetEvent(nextJob[i]);
			WaitForSingleObject(thds[i],INFINITE);
			myCloseHandle(thds[i]);
		}
	}

	for (i=TotalThreadsRequested-1; i>=0; i--)
	{
		myCloseHandle(nextJob[i]);
		myCloseHandle(jobFinished[i]);
	}

	TotalThreadsRequested=0;
	CurrentThreadsAllocated=0;
	CurrentThreadsUsed=0;
	*JobsRunning=false;
	*ThreadPoolRequested=false;
}


uint8_t ThreadPool::GetThreadNumber(uint8_t thread_number,bool logical)
{
	const uint8_t nCPU=(logical) ? CPU.NbLogicCPU:CPU.NbPhysCore;

	if (thread_number==0) return((nCPU>MAX_MT_THREADS) ? MAX_MT_THREADS:nCPU);
	else return(thread_number);
}


bool ThreadPool::AllocateThreads(uint8_t thread_number,uint8_t offset)
{
	if ((!Status_Ok) || (thread_number==0)) return(false);

	if (thread_number>CurrentThreadsAllocated)
	{
		TotalThreadsRequested=thread_number;
		CreateThreadPool(offset);
		if (!Status_Ok) return(false);
	}

	return(true);
}


bool ThreadPool::DeAllocateThreads(void)
{
	if (!Status_Ok) return(false);

	FreeThreadPool();

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
		jobFinished[i]=CreateEvent(NULL,TRUE,TRUE,NULL);
		nextJob[i]=CreateEvent(NULL,TRUE,FALSE,NULL);
		MT_Thread[i].jobFinished=jobFinished[i];
		MT_Thread[i].nextJob=nextJob[i];
		Status_Ok=Status_Ok && ((MT_Thread[i].jobFinished!=NULL) && (MT_Thread[i].nextJob!=NULL));
		i++;
	}
	if (!Status_Ok)
	{
		FreeThreadPool();
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
		return;
	}

	CurrentThreadsAllocated=TotalThreadsRequested;
}



bool ThreadPool::RequestThreadPool(uint8_t thread_number,Public_MT_Data_Thread *Data)
{
	if ((!Status_Ok) || (thread_number>CurrentThreadsAllocated)) return(false);

	for(uint8_t i=0; i<thread_number; i++)
		MT_Thread[i].MTData=Data+i;

	CurrentThreadsUsed=thread_number;

	*ThreadPoolRequested=true;
	ResetEvent(ThreadPoolFree);

	return(true);	
}


bool ThreadPool::ReleaseThreadPool(void)
{
	if (!Status_Ok) return(false);

	if (*ThreadPoolRequested)
	{
		for(uint8_t i=0; i<CurrentThreadsUsed; i++)
			MT_Thread[i].MTData=NULL;
		CurrentThreadsUsed=0;
		*ThreadPoolRequested=false;
		SetEvent(ThreadPoolFree);
	}

	return(true);
}


bool ThreadPool::StartThreads(void)
{
	if ((!Status_Ok) || (!*ThreadPoolRequested) || (CurrentThreadsUsed==0)) return(false);

	if (*JobsRunning) return(true);

	*JobsRunning=true;
	ResetEvent(JobsEnded);

	for(uint8_t i=0; i<CurrentThreadsUsed; i++)
	{
		MT_Thread[i].f_process=1;
		ResetEvent(jobFinished[i]);
		SetEvent(nextJob[i]);
	}

	return(true);	
}


bool ThreadPool::WaitThreadsEnd(void)
{
	if ((!Status_Ok) || (!*ThreadPoolRequested) || (CurrentThreadsUsed==0)) return(false);

	if (!*JobsRunning) return(true);

	WaitForMultipleObjects(CurrentThreadsUsed,jobFinished,TRUE,INFINITE);

	for(uint8_t i=0; i<CurrentThreadsUsed; i++)
		MT_Thread[i].f_process=0;

	*JobsRunning=false;
	SetEvent(JobsEnded);

	return(true);
}

