#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

/*
Linked List Structure
*/
 struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
 };

/*
Definition for a Pair
*/
class Pair {
public:
    int key;
    string value;

    Pair(int key, string value) : key(key), value(value) {}
};


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
Problem: Products of Array Except Self
Leet Code Link: https://leetcode.com/problems/product-of-array-except-self/description/
*/
vector<int> productExceptSelf(vector<int>& nums)
{
    int numsLength = nums.size();
    vector<int> output(numsLength, 1);

    for (int i = 1; i < numsLength; i++)
    {
        output[i] = output[i-1] * nums[i-1];
    }

    int postFix = 1;
    for (int i = (numsLength - 1); i >= 0; i--)
    {
        output[i] = postFix * output[i];
        postFix *= nums[i];
    }

    return output;
}

/*
Problem: Valid Sudoku
Leet Code Link: https://leetcode.com/problems/valid-sudoku/description/
*/
bool isValidSudoku(vector<vector<char>>& board)
{
    unordered_map<int, unordered_set<char>> rows, cols;
    map<pair<int, int>, unordered_set<char>> squares;

    for(int r = 0; r < 9; r++)
    {
        for (int c = 0; c < 9; c++)
        {
            if (board[r][c] == '.')
                continue;

            pair<int, int> squareKey = {r / 3, c / 3};

            if (rows[r].count(board[r][c]) || cols[c].count(board[r][c]) || squares[squareKey].count(board[r][c])) {
                return false;
            }

            rows[r].insert(board[r][c]);
            cols[c].insert(board[r][c]);
            squares[squareKey].insert(board[r][c]);
        }
    }

    return true;
}

/*
Problem: Longest Consecutive Sequence
Leet COde Link: https://leetcode.com/problems/longest-consecutive-sequence/
*/
int longestConsecutive(vector<int>& nums) {
    unordered_map<int, int> mapConsecutives;
    int res = 0;

    for (int num : nums)
    {
        if (!mapConsecutives[num]) {
            int previous = mapConsecutives[num - 1];
            int next = mapConsecutives[num + 1];

            mapConsecutives[num] = previous + next + 1;
            mapConsecutives[num - previous] = mapConsecutives[num];
            mapConsecutives[num + next] = mapConsecutives[num];
            res = max(res, mapConsecutives[num]);
        }
    }

    return res;
}

/*
Problem: Insertion Sort
Leet Code Link: Could Not Find
*/
vector<vector<pair<int, string>>> insertionSort(vector<pair<int, string>>& pairs)
{
    vector<vector<pair<int, string>>> output;
    int pairsLength = pairs.size();

    for (int i = 0; i < pairsLength; i++)
    {
        int j = i - 1;
        while (j >= 0 && pairs[j + 1].first < pairs[j].first)
        {
            pair temp = pairs[j + 1];
            pairs[j + 1] = pairs[j];
            pairs[j] = temp;
            j--;
        }

        output.push_back(pairs);
    }

    return output;
}

/*
Problem: Merge Sort
Leet Code Link:
*/
void merge(vector<pair<int,string>>& pairs, int start, int middle, int end)
{
    vector<pair<int,string>> leftSub = {pairs.begin() + start, pairs.begin() + middle + 1};
    vector<pair<int,string>> rightSub = {pairs.begin() + middle + 1, pairs.begin() + end + 1};

    int i = 0; // index for L
    int j = 0; // index for R
    int k = start; // index for arr

    while (i < leftSub.size() && j < rightSub.size())
    {
        if (leftSub[i].first <= rightSub[j].first)
        {
            pairs[k] = leftSub[i++];
        }
        else
        {
            pairs[k] = rightSub[j++];
        }
        k++;
    }

    while (i < leftSub.size())
    {
        pairs[k++] = leftSub[i++];
    }

    while (j < rightSub.size())
    {
        pairs[k++] = rightSub[j++];
    }
}

void mergeSortHelper(vector<pair<int,string>>& pairs, int start, int end)
{
    if (end-start+1 <= 1)
        return;

    int middle = (start+end)/2;
    mergeSortHelper(pairs,start,middle);
    mergeSortHelper(pairs,middle+1,end);

    merge(pairs,start,middle,end);
}

vector<pair<int,string>> mergeSort(vector<pair<int,string>>& pairs)
{
    mergeSortHelper(pairs, 0, pairs.size() - 1);
    return pairs;
}

/*
Problem: Merge K Sorted Linked Lists
Leet Code Link: https://leetcode.com/problems/merge-k-sorted-lists/
*/
ListNode* merge(ListNode* left, ListNode* right)
{
    ListNode dummy;
    ListNode* current = &dummy;

    while(left != nullptr && right != nullptr)
    {
        if (left->val <= right->val)
        {
            current->next = left;
            left = left->next;
        }
        else
        {
            current->next = right;
            right = right->next;
        }

        current = current->next;
    }

    if (left != nullptr)
    {
        current->next = left;
    }
    else if (right != nullptr){
        current->next = right;
    }

    return dummy.next;
}

ListNode* divide(vector<ListNode*>& lists,int start, int end)
{
    if (start > end)
    {
        return nullptr;
    }
    if (start == end)
    {
        return lists[start];
    }

    int middle = (start + end) / 2;

    ListNode* left = divide(lists,start,middle);
    ListNode* right = divide(lists,middle+1, end);

    return merge(left,right);
}

ListNode* mergeKLists(vector<ListNode*>& lists)
{
    int listsSize = lists.size() - 1;

    if (listsSize <= 0)
    {
        return nullptr;
    }

    return divide(lists,0,listsSize);
}

/*
Problem: Quick Sort
Leet Code Link: Could Not Find
*/
void quickSort(vector<Pair>& pairs,int start, int end)
{
    if (end - start <= 0)
        return;

    Pair pivot = pairs[end];
    int left = start;

    for (int i = start; i < end; i++)
    {
        if (pairs[i].key < pivot.key)
        {
            swap(pairs[i], pairs[left]);
            left++;
        }
    }

    pairs[end] = pairs[left];
    pairs[left] = pivot;

    quickSort(pairs,start,left - 1);
    quickSort(pairs,left + 1,end);
}

vector<Pair> quickSort(vector<Pair>& pairs)
{
    int pairsSize = pairs.size();
    quickSort(pairs,0,pairsSize - 1);
    return pairs;
}

/*
Problem: Sort Colours
Leet COde Link: https://leetcode.com/problems/sort-colors/
*/
void sortColors(vector<int>& nums) {
    int sizeNums = nums.size();
    int left = 0;
    int current = 0;
    int right = sizeNums - 1;

    while(current <= right)
    {
        if (nums[current] == 0)
        {
            swap(nums[current], nums[left]);
            left++;
            current++;
        }
        else if (nums[current] == 1)
        {
            current++;
        }
        else
        {
            swap(nums[current], nums[right]);
            right--;
        }
    }
}

/*
Problem: Binary Search
Leet Code Link: https://leetcode.com/problems/binary-search/description/
*/
int search(vector<int>& nums, int target)
{
    int sizeNums = nums.size();
    int low = 0;
    int high = sizeNums - 1;

    while (low < high)
    {
        int middle = low + (high - low) / 2;
        if (nums[middle] >= target)
        {
            high = middle;
        }
        else
        {
            low = middle + 1;
        }
    }

    return (low < sizeNums && nums[low] == target) ? low : -1;
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