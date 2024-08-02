#include<bits/stdc++.h>
using namespace std;

int main(){
    map<int,int> mp;
    for(int i=6;i<=200;i++){
    //for(int i=4;i<=130;i++){
        ifstream fin("D:\\VS_project\\view_space\\Tammes_sphere\\"+to_string(i)+".txt");
        //ifstream fin("D:\\VS_project\\view_space\\Sphere_Codes\\pack.3."+to_string(i)+".txt");
        int num,id;
        double dis,angel,x,y,z;
        fin>>num>>dis>>angel;
        if(num!=i) cout<<"?"<<endl;
        vector<vector<double>> pts;
        while(fin>>id>>x>>y>>z){
        //while(fin>>x>>y>>z){
            //if(z>=0){
            //    vector<double> pt;
            //    pt.push_back(x);
            //    pt.push_back(y);
            //    pt.push_back(z);
            //    pts.push_back(pt);
            //}
            //if(y>=0){
            //    vector<double> pt;
            //    pt.push_back(x);
            //    pt.push_back(z);
            //    pt.push_back(y);
            //    pts.push_back(pt);
            //}
            if(x>=0){
                vector<double> pt;
                pt.push_back(z);
                pt.push_back(y);
                pt.push_back(x);
                pts.push_back(pt);
            }
        }
        //cout<<pts.size()<<endl;
        mp[pts.size()]++;
        ifstream fcheck("D:\\VS_project\\view_space\\Hemisphere\\"+to_string(pts.size())+".txt");
        if(!fcheck.good()){
            ofstream fout("D:\\VS_project\\view_space\\Hemisphere\\"+to_string(pts.size())+".txt");
            for(int j=0;j<pts.size();j++)
                fout<<pts[j][0]<<' '<<pts[j][1]<<' '<<pts[j][2]<<'\n';
        }
    }
    for(int i=3;i<=100;i++)
    //for(int i=3;i<=65;i++)
        if(mp[i] == 0) cout<<i<<endl;
    return 0;
}
