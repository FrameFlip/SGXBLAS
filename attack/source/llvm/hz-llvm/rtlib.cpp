#include <string>
#include <iostream>
#include <map>

extern std::map<const char*, int> mapPointerTrue;
extern std::map<const char*, int> mapPointerFalse;


void getCondVal(int conVal, const char* conStr){
//printf("1\n");
//	std::cout <<"1\n";
	std::cout << conVal <<" ["<<conStr <<"]" << std::endl;
	if(conVal == 1) {
		if(mapPointerTrue.find(conStr) != mapPointerTrue.end()){
			mapPointerTrue[conStr]++;
		} else {
			mapPointerTrue[conStr]=1;
		}
		if(mapPointerFalse.find(conStr) == mapPointerFalse.end()){
			mapPointerFalse[conStr] = 0;
		}
	} else {
		if(mapPointerFalse.find(conStr) != mapPointerFalse.end()){
			mapPointerFalse[conStr]++;
		} else {
			mapPointerFalse[conStr]=1;
		}	
		if(mapPointerTrue.find(conStr) == mapPointerTrue.end()){
			mapPointerTrue[conStr] = 0;
		}
	}
/*	if(conVal == 1)printf("1\n");
	else printf("0\n");*/
}
