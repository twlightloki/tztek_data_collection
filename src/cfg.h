#pragma once
#include <string>
using namespace std;
int CFG_init(const char *filename);
int CFG_get_section_value(const char *filename,const char *section,const char *key,char *buf,int buflen);
int CFG_get_section_value_int(const char *filename,const char *section,const char *key,int defval=0);
long CFG_get_section_value_long(const char *filename,const char *section,const char *key,long defval=0);
float CFG_get_section_value_float(const char *filename,const char *section,const char *key,float defval=0);
string CFG_get_section_value_string(const char *filename,const char *section,const char *key,string defval="");
int CFG_free(const char *filename);

