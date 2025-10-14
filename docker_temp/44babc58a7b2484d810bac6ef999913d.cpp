# Write your Python code here
#include<bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string s; cin>>s;
    vector<int> ans;
    int count=0;
    for(int i=0; i<(int)s.size(); i++){
        if(s[i]==0) count++;
        else{
            ans.push_back(s[i]);
        }
    } 
    for(int i=0; i<count;i++) ans.push_back('0');

    for(int i=0; i<(int)ans.size() ; i++) cout<<ans[i]<<" ";
    cout<<endl;
    

    return 0;
}