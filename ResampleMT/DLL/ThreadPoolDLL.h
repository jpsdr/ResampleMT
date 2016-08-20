#ifdef THREADPOOLDLL_EXPORTS
#define THREADPOOLDLL_API __declspec(dllexport) 
#else
#define THREADPOOLDLL_API __declspec(dllimport) 
#endif

#include <stdint.h>

#define MAX_MT_THREADS 128

typedef void (*ThreadPoolFunction)(void *ptr);


typedef struct _Public_MT_Data_Thread
{
	ThreadPoolFunction pFunc;
	void *pClass;
	uint8_t f_process,thread_Id;
} Public_MT_Data_Thread;


namespace ThreadPoolDLL
{

class ThreadPoolInterface
{
	public:

	static THREADPOOLDLL_API uint8_t GetThreadNumber(uint8_t thread_number);
	static THREADPOOLDLL_API bool AllocateThreads(uint8_t thread_number);
	static THREADPOOLDLL_API bool RequestThreadPool(DWORD pId,uint8_t thread_number,Public_MT_Data_Thread *Data);
	static THREADPOOLDLL_API bool ReleaseThreadPool(DWORD pId);
	static THREADPOOLDLL_API bool StartThreads(DWORD pId);
	static THREADPOOLDLL_API bool WaitThreadsEnd(DWORD pId);
	static THREADPOOLDLL_API bool GetThreadPoolStatus(void);
	static THREADPOOLDLL_API uint8_t GetCurrentThreadAllocated(void);
	static THREADPOOLDLL_API uint8_t GetCurrentThreadUsed(void);
};

}