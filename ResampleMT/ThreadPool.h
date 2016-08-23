#ifndef __ThreadPool_H__
#define __ThreadPool_H__

#include <stdint.h>
#include <Windows.h>

#define MAX_MT_THREADS 128
#define MAX_THREAD_POOL 1

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

	virtual ~ThreadPool(void);
	static ThreadPool& Init(uint8_t num);

	protected :

	Arch_CPU CPU;

	public :

	uint8_t GetThreadNumber(uint8_t thread_number,bool logical);
	bool AllocateThreads(DWORD pId,uint8_t thread_number,uint8_t offset);
	bool DeAllocateThreads(DWORD pId);
	bool RequestThreadPool(DWORD pId,uint8_t thread_number,Public_MT_Data_Thread *Data);
	bool ReleaseThreadPool(DWORD pId);
	bool StartThreads(DWORD pId);
	bool WaitThreadsEnd(DWORD pId);
	bool GetThreadPoolStatus(void) {return(Status_Ok);}
	uint8_t GetCurrentThreadAllocated(void) {return(CurrentThreadsAllocated);}
	uint8_t GetCurrentThreadUsed(void) {return(CurrentThreadsUsed);}
	uint8_t GetLogicalCPUNumber(void) {return(CPU.NbLogicCPU);}
	uint8_t GetPhysicalCoreNumber(void) {return(CPU.NbPhysCore);}

	protected :

	ThreadPool(void);

	MT_Data_Thread MT_Thread[MAX_MT_THREADS];
	HANDLE thds[MAX_MT_THREADS];
	DWORD tids[MAX_MT_THREADS];
	ULONG_PTR ThreadMask[MAX_MT_THREADS];
	CRITICAL_SECTION CriticalSection;
	BOOL CSectionOk;
	HANDLE JobsEnded,ThreadPoolFree;
//	DWORD TabId[MAX_USERS];

	bool Status_Ok,ThreadPoolRequested,JobsRunning;
	uint8_t TotalThreadsRequested,CurrentThreadsAllocated,CurrentThreadsUsed;
	DWORD ThreadPoolRequestProcessId;
	uint16_t NbreUsers;
	
	void FreeData(void);
	void FreeThreadPool(void);
	void CreateThreadPool(uint8_t offset);

	private :

	static DWORD WINAPI StaticThreadpool(LPVOID lpParam);

	ThreadPool (const ThreadPool &other);
	ThreadPool& operator = (const ThreadPool &other);
	bool operator == (const ThreadPool &other) const;
	bool operator != (const ThreadPool &other) const;
};

#endif // __ThreadPool_H__
