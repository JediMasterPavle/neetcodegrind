#include <iostream>
#include <vector>
#include <set>

using namespace std;

/*
"Contains Duplicate" Leet Code Question: https://leetcode.com/problems/contains-duplicate/description/
*/
bool hasDuplicate(std::vector<int>& nums) {
        std::set<int> set;
        for (int& i : nums) {
            if (set.find(i) != set.end()) {
                return true;
            }
            set.insert(i);
        }
        return false;
}
/*
"Valid Anagram" Leet Code Question: https://leetcode.com/problems/valid-anagram/description/
*/
bool isAnagram(string s, string t) {
    
    if (s.length() != t.length())
        return false;

    vector<int> count(26,0);
    int length = s.length();
    for(int i = 0;i<length;i++){
        count[s[i]-'a'] +=1;
        count[t[i]-'a'] -=1;
    }

    for (int val : count) {
        if (val != 0) {
            return false;
        }
    }

    return true;
}

/*
This is the main function which doesn't do anything,
the functions/classes above will be answers to NeetCode questions
*/
int main(){
    cout<<"Neet Code Grind";
    return 0;
}