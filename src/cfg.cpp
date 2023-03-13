#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <string>
#include <map>
using namespace std;

typedef struct _cfg_des{
	char *data;
	int   size;
}cfg_des;

pthread_mutex_t g_map_lock = PTHREAD_MUTEX_INITIALIZER;
static map<string,cfg_des> g_map_cfg;

static int get_file_size(const char *filename)
{
	struct stat st;
	st.st_size = 0;
	stat(filename, &st);
	return st.st_size;
}

static int load_file(const char *filename,char **data,int *datalen)
{
	int filesize = 0;
	char *buf = NULL;
	int fd = -1;
	int retval = -1;

	do
	{
		if ((filesize = get_file_size(filename)) == 0){
		  printf("file size is zero\n");
		  break;
		}

		if((buf = (char*)malloc(filesize)) == NULL){
		  printf("malloc %d err,errno %d\n",filesize,errno);
		  break;
		}
		memset(buf,0,filesize);

		if ((fd = open(filename,O_RDONLY)) == -1){
		  printf("open %s err,errno %d\n",filename,errno);
		  break;
		}

		if (read(fd,buf,filesize) != filesize){
		  printf("read %s err,errno %d\n",filename,errno);
		  break;
		}

		retval = 0;
		*data = buf;
		*datalen = filesize;

	}while(0);


	if (fd != -1){
		close(fd);
		fd = -1;
	}

	if (retval ==-1){
		if (buf == NULL){
		  free(buf);
		  buf = NULL;
		}
	}
	return retval;
}


int tztek_get_line(const char **data,char *buf,int buflen)
{
	const char *src = *data;
	const char *pos1 = NULL;
	int linelen = 0;

	pos1 = strstr(src,"\n");
	if(pos1 != NULL){
		linelen = pos1 - src;
		*data += linelen+1;
		//remove \r
		if(linelen > 1 && *(pos1-1) == '\r'){
			linelen--;
		}
	}else{
		linelen = strlen(src);
		*data = NULL;
	}

	if(linelen > buflen){
		//warnning
		printf("linelen[%d] > buflen[%d]\n",linelen,buflen);
		linelen = buflen;
	}

	strncpy(buf,src,linelen);
	return 0;
}
 
int tztek_get_value(const char *data,const char *key,char *buf,int buflen)
{
	char buffer[4096] = {0};
	char *str1,*str2,*str3;
 
	while(data)
	{	
		memset(buffer,0,sizeof(buffer));
		if(tztek_get_line(&data,buffer,sizeof(buffer)-1) < 0){
			break;
		}
		
		//printf("%s:%s\n",__func__,buffer);
		str1 = buffer;	
 
		while( (' '==*str1) || ('\t'==*str1) ){
			str1++;
		}
		if( '#'==*str1 )	{
			continue;
		}
		if( ('/'==*str1)&&('/'==*(str1+1)) )	{
			continue;	
		}
		if( ('\0'==*str1)||(0x0d==*str1)||(0x0a==*str1) )	{
			continue;	
		}
		if( '['==*str1 ){
			/*continue*/;
			str2 = str1;
			while( (']'!=*str1)&&('\0'!=*str1) ){
				str1++;
			}
			if( ']'==*str1 ){
				break;
			}
			str1 = str2;
		}	
 
		str2 = str1;
 
		while( ('='!=*str1)&&('\0'!=*str1)&&(' '!=*str1)&&(':'!=*str1)){
			str1++;
		}
		if( '\0'==*str1 )	{
			continue;
		}
		str3 = str1+1;
 
		if( str2==str1 ){
			continue;	
		}
		*str1 = '\0';
 
		str1--;
 
		while( (' '==*str1)||('\t'==*str1) ){
			*str1 = '\0';
			str1--;
		}
 
		if( strcmp(str2,key) == 0){
			//printf("%s:%s\n",__func__,buffer);
			str1 = str3;
			while( (' '==*str1)||('\t'==*str1) ){
				str1++;
			}
			while( ('='==*str1)||(' '==*str1) ) {
				str1++;
			}
			str3 = str1;
 
			while( ('\0'!=*str1)&&(0x0d!=*str1)&&(0x0a!=*str1) ){
				if( ('/'==*str1)&&('/'==*(str1+1)) ){
					break;
				}
				str1++;	
			}	

			do{
				*str1 = '\0';
			}while(*--str1 == ' ');
			strncpy(buf,str3,buflen);
		 	return 0;
		}
	}
 
	return -1;
}

int CFG_init(const char *filename)
{
	char *data = NULL;
	int  datalen = 0;
	cfg_des des;
	int ret = -1;

	pthread_mutex_lock(&g_map_lock);

	do
	{
		if(g_map_cfg.find(filename) !=g_map_cfg.end()){
			printf("cfg alread init\n");
			break;
		}

		if(load_file(filename,&data,&datalen) < 0){
			printf("load file failed\n");
			break;
		}

		ret = 0;
		des.data= data;
		des.size = datalen;
		g_map_cfg[filename] = des;
		
	}while(0);
	
	pthread_mutex_unlock(&g_map_lock);
	return ret;
}

int CFG_free(const char *filename)
{
	pthread_mutex_lock(&g_map_lock);
	auto iter = g_map_cfg.find(filename);
	if(iter != g_map_cfg.end()){
		free(iter->second.data);
		g_map_cfg.erase(iter);
	}
	pthread_mutex_unlock(&g_map_lock);
	return 0;
}

int CFG_get_section_value(const char *filename,const char *section,const char *key,char *buf,int buflen)
{
	char buffer[4096] = {0};
	char *str1,*str2;
	int retval = -1;
	char *data = NULL;

	pthread_mutex_lock(&g_map_lock);
	auto iter = g_map_cfg.find(filename);
	if(iter == g_map_cfg.end()){
		return -1;
	}else{
		data = iter->second.data;
	}
	
	while(data){
		memset(buffer,0,sizeof(buffer));
		if(tztek_get_line((const char**)&data,buffer,sizeof(buffer)-1) < 0){
			break;
		}
		str1 = buffer;
 
		while( (' '==*str1) || ('\t'==*str1) ){
			str1++;
		}
		if( '['==*str1 ){
			str1++;
			while( (' '==*str1) || ('\t'==*str1) ){
				str1++;
			}
			str2 = str1;
 
			while( (']'!=*str1) && ('\0'!=*str1) ){
				str1++;
			}
			if( '\0'==*str1)	{
				continue;
			}
			while( ' '==*(str1-1) ){
				str1--;	
			}
			*str1 = '\0';

 			if(strcmp(str2,section) == 0){
				retval =  tztek_get_value(data,key,buf,buflen);
			}
 
		}					
	}
	pthread_mutex_unlock(&g_map_lock);
	return retval;
}

int CFG_get_section_value_int(const char *filename,const char *section,const char *key,int defval)
{
	char value[1024] = {0};
	if(CFG_get_section_value(filename,section,key,value,sizeof(value)-1) < 0){
		return defval;
	}
	return atoi(value);;
}

long CFG_get_section_value_long(const char *filename,const char *section,const char *key,long defval)
{
	char value[1024] = {0};
	if(CFG_get_section_value(filename,section,key,value,sizeof(value)-1) < 0)
	{
		return defval;
	}
	return atol(value);
}

float CFG_get_section_value_float(const char *filename,const char *section,const char *key,float defval)
{
	char value[1024] = {0};
	if(CFG_get_section_value(filename,section,key,value,sizeof(value)-1) < 0){
		return defval;
	}
	return atof(value);
}

string CFG_get_section_value_string(const char *filename,const char *section,const char *key,string defval)
{
	char value[1024] = {0};
	if(CFG_get_section_value(filename,section,key,value,sizeof(value)-1) < 0){
		return defval;
	}
	return value;
}

#if 0
int main(int argc,char *argv[])
{
	char *data = NULL;
	int  datalen = 0;
	char value[1024];

	if(cfg_init("config.ini") < 0){
		return -1;
	}
	
	// printf("value is %d\n", cfg_get_section_value("config.ini", "push", "push_rate"));

	CFG_get_section_value("config.ini", "push", "gps_server_ip", value, sizeof(value)-1);
	
	printf("value is %s\n", value);
	
	return 0;
}
#endif

