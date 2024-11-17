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
This is the main function which doesn't do anything,
the functions/classes above will be answers to NeetCode questions
*/
int main(){
    cout<<"Neet Code Grind";
    return 0;
}