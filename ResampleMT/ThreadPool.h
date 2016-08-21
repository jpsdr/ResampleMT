#ifndef __ThreadPool_H__
#define __ThreadPool_H__

#include <stdint.h>
#include <Windows.h>

#define MAX_MT_THREADS 128

typedef void (*ThreadPoolFunction)(void *ptr);


typedef struct _Public_MT_Data_Thread
{
	ThreadPoolFunction pFunc;
	void *pClass;
	uint8_t f_process,thread_Id;
} Public_MT_Data_Thread;



typedef struct _MT_Data_Thread
{
	Public_MT_Data_Thread *MTData;
	uint8_t f_process,thread_Id;
	HANDLE nextJob, jobFinished;
} MT_Data_Thread;


typedef struct _Arch_CPU
{
	uint8_t NbPhysCore,NbLogicCPU;
	uint8_t NbHT[64];
	ULONG_PTR ProcMask[64];
} Arch_CPU;


class ThreadPool
{
	public :

	ThreadPool(void);
	virtual ~ThreadPool(void);

	uint8_t GetThreadNumber(uint8_t thread_number);
	bool AllocateThreads(DWORD pId,uint8_t thread_number);
	bool DeAllocateThreads(DWORD pId);
	bool RequestThreadPool(DWORD pId,uint8_t thread_number,Public_MT_Data_Thread *Data);
	bool ReleaseThreadPool(DWORD pId);
	bool StartThreads(DWORD pId);
	bool WaitThreadsEnd(DWORD pId);
	bool GetThreadPoolStatus(void) {return(Status_Ok);}
	uint8_t GetCurrentThreadAllocated(void) {return(CurrentThreadsAllocated);}
	uint8_t GetCurrentThreadUsed(void) {return(CurrentThreadsUsed);}

	private :

	MT_Data_Thread MT_Thread[MAX_MT_THREADS];
	HANDLE thds[MAX_MT_THREADS];
	DWORD tids[MAX_MT_THREADS];
	Arch_CPU CPU;
	ULONG_PTR ThreadMask[MAX_MT_THREADS];
	HANDLE ghMutex,JobsEnded,ThreadPoolFree;
//	DWORD TabId[MAX_USERS];

	bool Status_Ok,ThreadPoolRequested,JobsRunning;
	uint8_t TotalThreadsRequested,CurrentThreadsAllocated,CurrentThreadsUsed;
	DWORD ThreadPoolRequestProcessId;
	uint16_t NbreUsers;
	
	static DWORD WINAPI StaticThreadpool(LPVOID lpParam);

	void FreeData(void);
	void FreeThreadPool(void);
	void CreateThreadPool(void);

};

#endif // __ThreadPool_H__
