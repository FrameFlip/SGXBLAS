
void getAns(int a,int b){
	int res=0;
	if(a>b)
		res = a;
	else
		res = b;
	
	if(a < b)
		res = res*a;
	else
		res = res*b;
}

