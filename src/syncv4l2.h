#ifndef _H_SYNCV4L2_H
#define _H_SYNCV4L2_H
#include <sys/time.h>
#include <stdint.h>

/*触发参数*/
typedef struct tagSYNCV4L2_TSyncPara
{
        int                     nMode;                            //GPS模式,0:内部GPS     1:外接GPS      2:室内模式
        int                     nVersion;                         //协议版本,传1即可
        int                     nInitTriggerSignal;       //标识是否需要初始触发信号
        int                     nSlave;                            //标识是否为从设备
        char                  szDevname[32];            //触发设备名称,比如/dev/ttyTHS4
        int                      nReset;                            //标识是否需要复位相机
        int                      res[4];                               //预留
}SYNCV4L2_TSyncPara;

/*相机参数*/
typedef struct tagSYNCV4L2_TCameraPara
{
    int                 nWidth;                                     //相机分辨率
    int                 nHeight;                                    //相机分辨率          
    char              szDevName[32];                     //相机设备名,比如/dev/video0
    int                 nTriggerFps;                            //触发帧率
    int                 nTriggerPluseWidth;             //触发脉宽, 单位us
    int                 nTriggerPluseOffset;            //触发偏移, 单位us
    int                 nRender;                                   //是否渲染，调试时使用
    int                 nTriggerOnly;                           //仅打开触发，不取图
    int                 res2[4];                                      //预留
}SYNCV4L2_TCameraPara;


/*事件类型*/
typedef enum tagSYNCV4L2_EEventType
{
    SYNCV4L2_TIMESTAMP_ERROR = -1,		//读时间戳异常
    SYNCV4L2_FRAME_LOSE = -2,			//丢帧
    SYNCV4L2_FRAME_ERROR = -3,			//帧错误
}SYNCV4L2_EEventType;


/*帧事件附加参数*/
typedef struct tagSYNCV4L2_TEventFrameHint
{
	int nChan;			//通道
	struct timespec stTime;		//时间戳
}SYNCV4L2_TEventFrameHint;

/*
        function:数据回调指针
        nChan:通道号
        stTime:时间戳
        nWidth:图像宽度
        nHeight:图像高度
        pData:图像数据
        nDatalen:数据长度
        pUserData:用户数据
*/
typedef void (* PSYNCV4L2_DataCallback)(int nChan,struct timespec stTime,int nWidth,int nHeight,unsigned char *pData,int nDatalen,void *pUserData);

/*
        function:事件回调指针
        nEvent: 事件类型，参考SYNCV4L2_EEventType
        pHint1: 事件附件参数1,不同事件类型附加参数不同，具体为
		SYNCV4L2_FRAME_ERROR，对应附加参数为SYNCV4L2_TEventFrameHint
		SYNCV4L2_FRAME_ERROR，对应附加参数为SYNCV4L2_TEventFrameHint
		其它,对应为nullptr
        pHint2: 事件附件参数2，预留
        pUserData:用户数据
*/
typedef void (*PSYNCV4L2_EventCallback)(int nEvent,void* pHint1,void *pHint2,void *pUserData);

/*
        function:初始化
        return:成功0，失败-1
*/
int SYNCV4L2_Init(SYNCV4L2_TSyncPara *pPara);

/*
        function:反初始化
        return:成功0，失败-1
*/
int SYNCV4L2_Release();

/*
        function:设置事件回调
        return:成功0，失败-1
*/
int SYNCV4L2_SetEventCallback(PSYNCV4L2_EventCallback eventCallback,void *pUserData);

/*
        function:打开视频通道
        nChan:通道号
        pPara: 相机参数
        return:成功0，失败-1
        备注：通道号从0开始
*/
 int SYNCV4L2_OpenCamera(int nChan,SYNCV4L2_TCameraPara *pPara);

 /*
        function:关闭视频通道
        return:成功0，失败-1
*/
 int SYNCV4L2_CloseCamera(int nChan);

 /*
        function:设置数据回调
        return:成功0，失败-1
*/
int SYNCV4L2_SetDataCallback(int nChan,PSYNCV4L2_DataCallback pCb,void *pUserData) ;      

/*
        function:开始采图
        return:成功0，失败-1
*/
int SYNCV4L2_Start();

/*
        function:停止采图
        return:成功0，失败-1
*/
int SYNCV4L2_Stop();
                                                                                                    
#endif
