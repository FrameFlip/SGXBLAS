//
// Created by xmhuangzhen on 2021/12/9.
//

#include <iostream>
#include <string>
#include <map>


//static std::map* <BranchInst,int> mapPointer = null;

//static std::map <int,int> *mapPointer = NULL;

//std::map<BranchInst, int>* mapPointer = new std::map<BranchInst, int>() ;


extern void getCondVal(int,const char*);


std::map<const char*, int> mapPointerTrue/* = new std::map<const char*, int>() */;
std::map<const char*, int> mapPointerFalse /*= new std::map<const char*, int>()*/ ;


extern void getAns(int,int);

int main(){
    for(int TEST_NUM = 1; TEST_NUM <= 10; ++TEST_NUM){
    	 getAns(TEST_NUM+5,TEST_NUM);
    }
    
    
    std::cout <<"---True---\n";
    std::map<const char*,int>::iterator it1 = mapPointerTrue.begin();
    while(it1 != mapPointerTrue.end()){
	std::cout <<"[" << it1->first <<"]"<< it1->second<<"\n";
	it1++;    
    }

    std::cout <<"---False---\n";
    std::map<const char*,int>::iterator it2 = mapPointerFalse.begin();
    while(it2 != mapPointerFalse.end()){
	std::cout <<"[" << it2->first <<"]"<< it2->second<<"\n";
	it2++;    
    }
    std::cout<<"---------\n";

    return 0;
}
