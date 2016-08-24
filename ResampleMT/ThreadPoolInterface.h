#ifndef __ThreadPoolInterface_H__
#define __ThreadPoolInterface_H__

#include <Windows.h>

#include "ThreadPoolDef.h"

typedef struct _UserData
{
	uint16_t UserId;
	int8_t nPool;
} UserData;


class ThreadPoolInterface
{
	public :

	virtual ~ThreadPoolInterface(void);
	static ThreadPoolInterface& Init(uint8_t num);

	public :

	uint8_t GetThreadNumber(uint8_t thread_number,bool logical);
	bool AllocateThreads(uint16_t &UserId,uint8_t thread_number,uint8_t offset,int16_t nPool);
	bool DeAllocateThreads(uint16_t UserId);
	bool RequestThreadPool(uint16_t UserId,uint8_t thread_number,Public_MT_Data_Thread *Data,int16_t nPool);
	bool ReleaseThreadPool(uint16_t UserId);
	bool StartThreads(uint16_t UserId);
	bool WaitThreadsEnd(uint16_t UserId);
	bool GetThreadPoolStatus(uint16_t UserId,int16_t nPool);
	uint8_t GetCurrentThreadAllocated(uint16_t UserId,int16_t nPool);
	uint8_t GetCurrentThreadUsed(uint16_t UserId,int16_t nPool);
	uint8_t GetLogicalCPUNumber(void);
	uint8_t GetPhysicalCoreNumber(void);

	bool GetThreadPoolInterfaceStatus(void) {return(Status_Ok);}
	uint16_t GetCurrentPoolCreated(void) {return(NbrePool);}

	protected :

	ThreadPoolInterface(void);

	CRITICAL_SECTION CriticalSection;
	BOOL CSectionOk;
	HANDLE JobsEnded[MAX_THREAD_POOL],ThreadPoolFree[MAX_THREAD_POOL];
	UserData TabId[MAX_USERS];
	uint16_t NbreUsers;

	bool Status_Ok;
	bool ThreadPoolRequested[MAX_THREAD_POOL],JobsRunning[MAX_THREAD_POOL];
	uint8_t NbrePool,NbrePoolEvent;
	int16_t ThreadPoolRequestUserIndex;

	bool CreatePoolEvent(uint8_t num);
	void FreeData(void);
	void FreePool(void);
	bool EnterCS(void);
	void LeaveCS(void);
	
	private :

	ThreadPoolInterface (const ThreadPoolInterface &other);
	ThreadPoolInterface& operator = (const ThreadPoolInterface &other);
	bool operator == (const ThreadPoolInterface &other) const;
	bool operator != (const ThreadPoolInterface &other) const;
};

#endif // __ThreadPoolInterface_H__

