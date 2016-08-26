// ThreadPoolDLL.cpp : définit les fonctions exportées pour l'application DLL.
//

#include "ThreadPoolInterface.h"
#include "ThreadPool.h"

#define myfree(ptr) if (ptr!=NULL) { free(ptr); ptr=NULL;}
#define myCloseHandle(ptr) if (ptr!=NULL) { CloseHandle(ptr); ptr=NULL;}
#define mydelete(ptr) if (ptr!=NULL) { delete ptr; ptr=NULL;}


static ThreadPool *ptrPool[MAX_THREAD_POOL]={NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};


ThreadPoolInterface* ThreadPoolInterface::Init(uint8_t num)
{

	static ThreadPoolInterface PoolInterface;

	if (PoolInterface.EnterCS())
	{
		if (num>=MAX_THREAD_POOL) num=MAX_THREAD_POOL;

		if (num>PoolInterface.NbrePool)
		{
			if (PoolInterface.CreatePoolEvent(num))
			{
				while ((PoolInterface.NbrePool<num) && PoolInterface.Status_Ok)
				{
					ptrPool[PoolInterface.NbrePool]= new
						ThreadPool(PoolInterface.JobsEnded[PoolInterface.NbrePool],
							PoolInterface.ThreadPoolFree[PoolInterface.NbrePool],
							&PoolInterface.ThreadPoolRequested[PoolInterface.NbrePool],
							&PoolInterface.JobsRunning[PoolInterface.NbrePool]);
					PoolInterface.Status_Ok=PoolInterface.Status_Ok &&
						(ptrPool[PoolInterface.NbrePool]!=NULL);
					PoolInterface.NbrePool++;
				}
				if (!PoolInterface.Status_Ok)
				{
					PoolInterface.FreePool();
					PoolInterface.LeaveCS();
					PoolInterface.FreeData();
				}
				else PoolInterface.LeaveCS();
			}
			else
			{
				PoolInterface.FreePool();
				PoolInterface.LeaveCS();
				PoolInterface.FreeData();
			}
		}
		else PoolInterface.LeaveCS();
	}

	return(&PoolInterface);
}


bool ThreadPoolInterface::EnterCS(void)
{
	if ((!Status_Ok) || (CSectionOk==FALSE)) return(false);

	EnterCriticalSection(&CriticalSection);

	return(true);
}


void ThreadPoolInterface::LeaveCS(void)
{
	if (CSectionOk==TRUE) LeaveCriticalSection(&CriticalSection);
}


bool ThreadPoolInterface::CreatePoolEvent(uint8_t num)
{
	if ((!Status_Ok) || (num==0))  return(false);

	if (num>NbrePoolEvent)
	{
		while((NbrePoolEvent<num) && Status_Ok)
		{
			JobsEnded[NbrePoolEvent]=CreateEvent(NULL,TRUE,TRUE,NULL);
			ThreadPoolFree[NbrePoolEvent]=CreateEvent(NULL,TRUE,TRUE,NULL);
			Status_Ok=Status_Ok &&
				(JobsEnded[NbrePoolEvent]!=NULL) && (ThreadPoolFree[NbrePoolEvent]!=NULL);
			NbrePoolEvent++;
		}
		if (!Status_Ok) return(false);
	}

	return(true);
}


void ThreadPoolInterface::FreeData(void)
{
	int16_t i;

	if (NbrePool>0)
	{
		for(i=NbrePool-1; i>=0; i--)
			mydelete(ptrPool[i]);
		NbrePool=0;
	}

	if (NbrePoolEvent>0)
	{
		for(i=NbrePoolEvent-1; i>=0; i--)
		{
			myCloseHandle(ThreadPoolFree[i]);
			myCloseHandle(JobsEnded[i]);
		}
		NbrePoolEvent=0;
	}

	myCloseHandle(EndExclusive);
	if (CSectionOk==TRUE)
	{
		CSectionOk=FALSE;
		DeleteCriticalSection(&CriticalSection);
	}
}


void ThreadPoolInterface::FreePool(void)
{
	if (NbrePool>0)
	{
		for(int16_t i=NbrePool-1; i>=0; i--)
		{
			if (ptrPool[i]!=NULL)
			{
				while(ThreadPoolRequested[i])
				{
					LeaveCriticalSection(&CriticalSection);
					WaitForSingleObject(ThreadPoolFree[i],INFINITE);
					EnterCriticalSection(&CriticalSection);
				}
				ptrPool[i]->DeAllocateThreads();
			}
		}
	}
}


ThreadPoolInterface::ThreadPoolInterface(void):CSectionOk(FALSE),Status_Ok(true),NbrePool(0),NbrePoolEvent(0)
{
	CSectionOk=InitializeCriticalSectionAndSpinCount(&CriticalSection,0x00000040);
	if (CSectionOk==FALSE) Status_Ok=false;

	EndExclusive=NULL;
	ExclusiveMode=false;

	EndExclusive=CreateEvent(NULL,TRUE,TRUE,NULL);
	if (EndExclusive==NULL) Status_Ok=false;

	uint16_t i;

	for (i=0; i<MAX_THREAD_POOL; i++)
	{
		JobsEnded[i]=NULL;
		ThreadPoolFree[i]=NULL;
		ThreadPoolRequested[i]=false;
		JobsRunning[i]=false;
	}

	for(i=0; i<MAX_USERS; i++)
	{
		TabId[i].UserId=0;
		TabId[i].nPool=-1;
	}
}


ThreadPoolInterface::~ThreadPoolInterface(void)
{
	Status_Ok=false;
	FreeData();
}


uint8_t ThreadPoolInterface::GetThreadNumber(uint8_t thread_number,bool logical)
{
	if ((!Status_Ok) || (NbrePool==0)) return(0);
	else return(ptrPool[0]->GetThreadNumber(thread_number,logical));
}



bool ThreadPoolInterface::AllocateThreads(uint16_t &UserId,uint8_t thread_number,uint8_t offset_core,uint8_t offset_ht,bool UseMaxPhysCore,int8_t nPool)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if ((thread_number==0) || ((UserId==0) && (NbreUsers>=MAX_USERS)) || (nPool>=(int8_t)NbrePool) || (nPool<-1))
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (UserId==0)
	{
		uint16_t user=1;

		if (NbreUsers>0)
		{
			bool found=false;

			while (!found && (user<=NbreUsers))
			{
				uint16_t i=0;
				bool search=true;

				while (search && (i<NbreUsers))
					search=search && (TabId[i++].UserId!=user);
				found=search;
				if (!found) user++;
			}
		}
		NbreUsers++;
		UserId=user;
		TabId[NbreUsers-1].UserId=UserId;
		TabId[NbreUsers-1].nPool=-1;
	}
	else
	{
		uint16_t i=0;

		while ((NbreUsers>i) && (TabId[i].UserId!=UserId)) i++;
		if (i==NbreUsers)
		{
			LeaveCriticalSection(&CriticalSection);
			return(false);
		}
	}

	if (nPool==-1)
	{
		for(uint8_t i=0; i<NbrePool; i++)
		{
			if (thread_number>ptrPool[i]->GetCurrentThreadAllocated())
			{
				while (JobsRunning[i])
				{
					LeaveCriticalSection(&CriticalSection);
					WaitForSingleObject(JobsEnded[i],INFINITE);
					EnterCriticalSection(&CriticalSection);
				}
				Status_Ok=ptrPool[i]->AllocateThreads(thread_number,offset_core,offset_ht,UseMaxPhysCore);
				if (!Status_Ok)
				{
					FreePool();
					LeaveCriticalSection(&CriticalSection);
					FreeData();
					return(false);
				}
			}
		}
	}
	else
	{
		if (thread_number>ptrPool[nPool]->GetCurrentThreadAllocated())
		{
			while (JobsRunning[nPool])
			{
				LeaveCriticalSection(&CriticalSection);
				WaitForSingleObject(JobsEnded[nPool],INFINITE);
				EnterCriticalSection(&CriticalSection);
			}
			Status_Ok=ptrPool[nPool]->AllocateThreads(thread_number,offset_core,offset_ht,UseMaxPhysCore);
			if (!Status_Ok)
			{
				FreePool();
				LeaveCriticalSection(&CriticalSection);
				FreeData();
				return(false);
			}
		}
	}

	LeaveCriticalSection(&CriticalSection);

	return(true);
}


bool ThreadPoolInterface::DeAllocateThreads(uint16_t UserId)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if (NbreUsers==0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(true);
	}

	uint16_t index=0;

	while ((NbreUsers>index) && (TabId[index].UserId!=UserId)) index++;
	if (index==NbreUsers)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (index<NbreUsers-1)
	{
		for(uint16_t j=index+1; j<NbreUsers; j++)
			TabId[j-1]=TabId[j];
	}
	NbreUsers--;
	TabId[NbreUsers].UserId=0;
	TabId[NbreUsers].nPool=-1;

	if (NbreUsers==0) FreePool();

	LeaveCriticalSection(&CriticalSection);

	return(true);
}


bool ThreadPoolInterface::RequestThreadPool(uint16_t UserId,uint8_t thread_number,Public_MT_Data_Thread *Data,int8_t nPool,bool Exclusive)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if ((UserId==0) || (NbreUsers==0) || (nPool>=(int8_t)NbrePool) || (nPool<-1) || (thread_number==0) || (Data==NULL))
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	uint16_t userindex=0;

	while ((NbreUsers>userindex) && (TabId[userindex].UserId!=UserId)) userindex++;
	if (userindex==NbreUsers)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (TabId[userindex].nPool>=0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	while (ExclusiveMode)
	{
		LeaveCriticalSection(&CriticalSection);
		WaitForSingleObject(EndExclusive,INFINITE);
		EnterCriticalSection(&CriticalSection);
	}

	if (nPool==-1)
	{
		if ((NbrePool>1) && (!Exclusive))
		{
			DWORD a;
			bool PoolFree=false;

			while (!PoolFree)
			{
				LeaveCriticalSection(&CriticalSection);
				a=WaitForMultipleObjects(NbrePool,ThreadPoolFree,FALSE,INFINITE);
				EnterCriticalSection(&CriticalSection);
				nPool=(int8_t)(a-WAIT_OBJECT_0);
				PoolFree=!ThreadPoolRequested[nPool];
			}
		}
		else nPool=0;
	}

	if (thread_number>ptrPool[nPool]->GetCurrentThreadAllocated())
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if ((!Exclusive) || (NbrePool==1))
	{
		while (ThreadPoolRequested[nPool])
		{
			LeaveCriticalSection(&CriticalSection);
			WaitForSingleObject(ThreadPoolFree[nPool],INFINITE);
			EnterCriticalSection(&CriticalSection);
		}
	}
	else
	{
		bool PoolFree=false;

		while (!PoolFree)
		{
			LeaveCriticalSection(&CriticalSection);
			WaitForMultipleObjects(NbrePool,ThreadPoolFree,TRUE,INFINITE);
			EnterCriticalSection(&CriticalSection);
			PoolFree=true;
			for(uint8_t i=0; i<NbrePool; i++)
				PoolFree=PoolFree && (!ThreadPoolRequested[i]);
		}
	}

	bool out=ptrPool[nPool]->RequestThreadPool(thread_number,Data);

	if (out)
	{
		if (Exclusive)
		{
			ExclusiveMode=true;
			ResetEvent(EndExclusive);
		}
		TabId[userindex].nPool=(int8_t)nPool;
	}

	LeaveCriticalSection(&CriticalSection);

	return(out);	
}


bool ThreadPoolInterface::ReleaseThreadPool(uint16_t UserId)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if (UserId==0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	uint16_t index=0;

	while ((NbreUsers>index) && (TabId[index].UserId!=UserId)) index++;
	if (index==NbreUsers)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (TabId[index].nPool<0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(true);
	}

	uint8_t nPool=(uint8_t)(TabId[index].nPool);

	if (nPool>=NbrePool)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	bool out=true;

	if (ThreadPoolRequested[nPool])
	{
		while (JobsRunning[nPool])
		{
			LeaveCriticalSection(&CriticalSection);
			WaitForSingleObject(JobsEnded[nPool],INFINITE);
			EnterCriticalSection(&CriticalSection);
		}
		out=ptrPool[nPool]->ReleaseThreadPool();
		TabId[index].nPool=-1;
	}

	if (ExclusiveMode)
	{
		ExclusiveMode=false;
		SetEvent(EndExclusive);
	}

	LeaveCriticalSection(&CriticalSection);

	return(out);
}


bool ThreadPoolInterface::StartThreads(uint16_t UserId)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if (UserId==0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	uint16_t index=0;

	while ((NbreUsers>index) && (TabId[index].UserId!=UserId)) index++;
	if (index==NbreUsers)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (TabId[index].nPool<0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	uint8_t nPool=(uint8_t)(TabId[index].nPool);

	if (nPool>=NbrePool)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (!ThreadPoolRequested[nPool])
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (JobsRunning[nPool])
	{
		LeaveCriticalSection(&CriticalSection);
		return(true);
	}

	bool out=ptrPool[nPool]->StartThreads();

	LeaveCriticalSection(&CriticalSection);

	return(out);	
}


bool ThreadPoolInterface::WaitThreadsEnd(uint16_t UserId)
{
	if (!Status_Ok) return(false);

	EnterCriticalSection(&CriticalSection);

	if (UserId==0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	uint16_t index=0;

	while ((NbreUsers>index) && (TabId[index].UserId!=UserId)) index++;
	if (index==NbreUsers)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (TabId[index].nPool<0)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	uint8_t nPool=(uint8_t)(TabId[index].nPool);

	if (nPool>=NbrePool)
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (!ThreadPoolRequested[nPool])
	{
		LeaveCriticalSection(&CriticalSection);
		return(false);
	}

	if (!JobsRunning[nPool])
	{
		LeaveCriticalSection(&CriticalSection);
		return(true);
	}

	bool out=ptrPool[nPool]->WaitThreadsEnd();

	LeaveCriticalSection(&CriticalSection);

	return(out);
}


bool ThreadPoolInterface::GetThreadPoolStatus(uint16_t UserId,int8_t nPool)
{
	if (Status_Ok && (NbrePool>0))
	{
		if ((UserId==0) || (NbreUsers==0))
		{
			if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetThreadPoolStatus());
			else return(false);
		}
		else
		{
			uint16_t i=0;

			while ((NbreUsers>i) && (TabId[i].UserId!=UserId)) i++;
			if (i==NbreUsers)
			{
				if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetThreadPoolStatus());
				else return(false);
			}
			if ((TabId[i].nPool>=0) && (TabId[i].nPool<(int8_t)NbrePool))
				return(ptrPool[TabId[i].nPool]->GetThreadPoolStatus());
			else return(false);		
		}
	}
	else return(false);
}


uint8_t ThreadPoolInterface::GetCurrentThreadAllocated(uint16_t UserId,int8_t nPool)
{
	if (Status_Ok && (NbrePool>0))
	{
		if ((UserId==0) || (NbreUsers==0))
		{
			if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadAllocated());
			else return(0);
		}
		else
		{
			uint16_t i=0;

			while ((NbreUsers>i) && (TabId[i].UserId!=UserId)) i++;
			if (i==NbreUsers)
			{
				if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadAllocated());
				else return(0);
			}
			if ((TabId[i].nPool>=0) && (TabId[i].nPool<(int8_t)NbrePool))
				return(ptrPool[TabId[i].nPool]->GetCurrentThreadAllocated());
			else return(0);		
		}
	}
	else return(0);
}


uint8_t ThreadPoolInterface::GetCurrentThreadUsed(uint16_t UserId,int8_t nPool)
{
	if (Status_Ok && (NbrePool>0))
	{
		if ((UserId==0) || (NbreUsers==0))
		{
			if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadUsed());
			else return(0);
		}
		else
		{
			uint16_t i=0;

			while ((NbreUsers>i) && (TabId[i].UserId!=UserId)) i++;
			if (i==NbreUsers)
			{
				if ((nPool>=0) && (nPool<(int8_t)NbrePool)) return(ptrPool[nPool]->GetCurrentThreadUsed());
				else return(0);
			}
			if ((TabId[i].nPool>=0) && (TabId[i].nPool<(int8_t)NbrePool))
				return(ptrPool[TabId[i].nPool]->GetCurrentThreadUsed());
			else return(0);		
		}
	}
	else return(0);
}


uint8_t ThreadPoolInterface::GetLogicalCPUNumber(void)
{
	if (Status_Ok && (NbrePool>0)) return(ptrPool[0]->GetLogicalCPUNumber());
	else return(0);
}


uint8_t ThreadPoolInterface::GetPhysicalCoreNumber(void)
{
	if (Status_Ok && (NbrePool>0)) return(ptrPool[0]->GetPhysicalCoreNumber());
	else return(0);
}

