#include <iostream>
#include <unordered_map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

/*
Problem: Contains Duplicate
Leet Code Question: https://leetcode.com/problems/contains-duplicate/description/
*/
bool hasDuplicate(std::vector<int>& nums)
{
        std::set<int> set;
        for (int& i : nums)
        {
            if (set.find(i) != set.end())
            {
                return true;
            }
            set.insert(i);
        }

        return false;
}

/*
Problem: Valid Anagram
Leet Code Link: https://leetcode.com/problems/valid-anagram/description/
*/
bool isAnagram(string s, string t)
{
    if (s.length() != t.length())
        return false;

    vector<int> count(26,0);
    int length = s.length();

    for(int i = 0;i<length;i++)
    {
        count[s[i]-'a'] +=1;
        count[t[i]-'a'] -=1;
    }

    for (int val : count)
    {
        if (val != 0) {
            return false;
        }
    }

    return true;
}

/*
Problem: Group Anagrams
Leet Code Link: https://leetcode.com/problems/group-anagrams/description/
*/
vector<vector<string>> groupAnagrams(vector<string>& strs)
{
    unordered_map<string, vector<string>> store_anagrams;
    for (const auto& s : strs)
    {
        vector<int> count(26, 0);
        for (char c : s)
        {
            count[c - 'a']++;
        }

        int countSize = count.size();
        string key = to_string(count[0]);
        for (int i = 1; i < countSize; i++)
        {
            key += "#" + to_string(count[i]);
        }

        store_anagrams[key].push_back(s);
    }

    vector<vector<string>> result;
    for (auto& pair : store_anagrams)
    {
        result.push_back(pair.second);
    }

    return result;
}

/*
Problem: Two Sum
Leet Code Link: https://leetcode.com/problems/two-sum/description/
*/
vector<int> twoSum(vector<int>& nums, int target)
{
    if (nums.size() == 0)
        return {};

    std::unordered_map<int, int> twoSumMap;
    int numsSize = nums.size();
    for (int i=0; i < numsSize; i++)
    {
        int complement = target - nums[i];
        if (twoSumMap.find(complement) != twoSumMap.end())
            return {twoSumMap[complement],i};

        twoSumMap[nums[i]] = i;
    }

    return {};
}

/*
Problem: Top K Frequent Elements
Leet Code Link: https://leetcode.com/problems/top-k-frequent-elements/
*/
vector<int> topKFrequent(vector<int>& nums, int k)
{
    unordered_map<int, int> count;
    vector<vector<int>> freq(nums.size() + 1);

    for (int n : nums)
    {
        count[n] = 1 + count[n];
    }

    for (const auto& entry : count)
    {
        freq[entry.second].push_back(entry.first);
    }

    vector<int> res;
    for (int i = freq.size() - 1; i > 0; --i)
    {
        for (int n : freq[i])
        {
            res.push_back(n);
            if (res.size() == k)
            {
                return res;
            }
        }
    }
    return res;
}

/*
Problem: Encode and Decode Strings
Leet Code Link: https://leetcode.com/problems/encode-and-decode-strings/description/
*/
string encode(vector<string>& strs)
{
    string res = "";
    for (const string& str : strs)
    {
        res += str + '\n';
    }

    return res;
}

vector<string> decode(string s)
{
    vector<string> result;
    stringstream ss(s);
    string temp;

    while (getline(ss, temp))
    {
        result.push_back(temp);
    }

    return result;
}



/*
This is the main function which doesn't do anything,
the functions/classes above will be answers to NeetCode questions
*/
int main()
{
    cout<<"Neet Code Grind";
    return 0;
}