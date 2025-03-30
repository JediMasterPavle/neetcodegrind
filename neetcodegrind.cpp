#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

/*
Tree Node Structure
*/
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

/*
Tree Node Structure with (key,pair)
*/
struct TreeNode2
{
    int key;
    int value;
    TreeNode2* left;
    TreeNode2* right;

    TreeNode2(int k, int v) : key(k), value(v), left(nullptr), right(nullptr) {}
};

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
Trie Node Structure
*/
struct TrieNode
{
    public:
        unordered_map<char, TrieNode*> children;
        bool word = false;
};

class TrieNode2 {
    public:
        vector<TrieNode2*> children;
        bool word;

        TrieNode2() : children(26, nullptr), word(false) {}
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
Guess Random Number
*/
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> dis(1, INT32_MAX);
int randomNum = dis(gen);

int guess(int num)
{
    if (num > randomNum)
    {
        return -1;
    }
    else if (num < randomNum)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*
Checks if Random Number is Same as Parameter
*/
int isBadVersion(int guess)
{
    return guess == randomNum;
}

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
Problem: Search a 2D Matrix
Leet Code Link: https://leetcode.com/problems/search-a-2d-matrix/description/
*/
bool searchMatrix(vector<vector<int>>& matrix, int target)
{
    int rows = matrix.size();
    int cols = matrix[0].size();

    int left = 0;
    int right = (rows * cols) - 1;

    while (left <= right)
    {
        int middle = left + (right - left)/2;
        int row = middle / cols;
        int col = middle % cols;

        if (matrix[row][col] < target)
        {
            left = middle + 1;
        }
        else if (matrix[row][col] > target)
        {
            right = middle - 1;
        }
        else
        {
            return true;
        }
    }

    return false;
}

/*
Problem: Guess Number Higher or Lower
Leet Code Link: https://leetcode.com/problems/guess-number-higher-or-lower/description/
*/
int guessNumber(int n)
{
    int low = 1;
    int high = n;

    while (low <= high)
    {
        int middle = low + (high - low)/2;
        if (guess(middle) == -1)
        {
            high = middle - 1;
        }
        else if (guess(middle) == 1)
        {
            low = middle + 1;
        }
        else
        {
            return middle;
        }
    }

return 0;
}

/*
Problem: First Bad Version
Leet Code Link: https://leetcode.com/problems/first-bad-version/description/
*/
int firstBadVersion(int n)
{
    int low = 0;
    int high = n;
    int answer = n;

    while (low < high)
    {
        int middle = low + (high - low) / 2;
        if (isBadVersion(middle))
        {
            answer = middle;
            high = middle-1;
        }
        else
        {
            low = middle + 1;
        }
    }

    return answer;
}

/*
Problem: Koko Eating Bananas
Leet Code Link: https://leetcode.com/problems/koko-eating-bananas/description/
*/
int minEatingSpeed(vector<int>& piles, int h)
{
    int low = 1;
    int high = *max_element(piles.begin(), piles.end());
    int result = high;

    while (low <= high)
    {
        int k = low + (high - low)/2;

        int totalHours = 0;
        for (int pile : piles)
        {
            totalHours += ceil(static_cast<double>(pile) / k);
        }

        if (totalHours <= h)
        {
            result = k;
            high = k - 1;
        }
        else
        {
            low = k + 1;
        }
    }

    return result;
}

/*
Problem: Search in a Binary Search Tree
Leet Code Link: https://leetcode.com/problems/search-in-a-binary-search-tree/description/
*/
TreeNode* searchBST(TreeNode* root, int val)
{
    if (root == nullptr)
        return nullptr;

    if (val > root->val)
    {
        return searchBST(root->right, val);
    }
    else if (val < root->val)
    {
        return searchBST(root->left, val);
    }
    else
    {
        return root;
    }
}

/*
Problem: Insert into a Binary Search Tree
Leet Code Link: https://leetcode.com/problems/insert-into-a-binary-search-tree/description/
*/
TreeNode* insertIntoBST(TreeNode* root, int val)
{
    if (root == nullptr)
    {
        return new TreeNode(val);
    }

    if (val < root->val)
    {
        root->left = insertIntoBST(root->left, val);
    }
    else if (val > root->val)
    {
        root->right = insertIntoBST(root->right, val);
    }

    return root;
}

/*
Problem: Delete Node in a BST
Leet Code Link: https://leetcode.com/problems/delete-node-in-a-bst/description/
*/
TreeNode* FindMinValueNode(TreeNode* tree)
{
    TreeNode* current = tree;
    while (current != nullptr && current->left != nullptr)
    {
        current = current->left;
    }

    return current;
}

TreeNode* deleteNode(TreeNode* root, int key)
{
    if (root == nullptr)
        return nullptr;

    if (root->val < key)
    {
        root->left = deleteNode(root->left, key);
    }
    else if (root->val > key)
    {
        root->right = deleteNode(root->right, key);
    }
    else
    {
        if (root->left == nullptr)
        {
            return root->right;
        }
        else if (root->right == nullptr)
        {
            return root->left;
        }
        else
        {
            TreeNode* minNode = FindMinValueNode(root->right);
            root->val = minNode->val;
            root->right = deleteNode(root->right, minNode->val);
        }
    }

    return root;
}

/*
Problem: Binary Tree Inorder Traversal
Leet Code Link: https://leetcode.com/problems/binary-tree-inorder-traversal/description/
*/
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> output;
    if (root == nullptr)
    {
        return output;
    }

    vector<int> left = inorderTraversal(root->left);
    output.insert(output.end(), left.begin(),left.end());

    output.push_back(root->val);

    vector<int> right = inorderTraversal(root->right);
    output.insert(output.end(), right.begin(),right.end());

    return output;
}

/*
Problem: Kth Smallest Integer in BST
Leet Code Link: https://leetcode.com/problems/kth-smallest-element-in-a-bst/
*/
void inOrderTraversal(TreeNode* root, int& count, int& answer, int k)
{
    if(root == nullptr)
        return;

    inOrderTraversal(root->left, count, answer, k);
    count++;

    if(count == k)
    {
        answer = root->val;
        return;
    }

    inOrderTraversal(root->right, count, answer, k);
}

int countNodes(TreeNode* root)
{
    if (root == nullptr)
        return 0;

    int leftCount = countNodes(root->left);
    int rightCount = countNodes(root->right);

    return 1 + leftCount + rightCount;
}

int kthSmallest(TreeNode* root, int k)
{
    int answer;
    int count = 0;

    if (k > countNodes(root))
        return -1;

    inOrderTraversal(root, count, answer, k);
    return answer;
}

/*
Problem: Construct Binary Tree from Preorder and Inorder Traversal
Leet Code Link: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/
*/
TreeNode* buildTreeHelper(vector<int>& preorder, int preStart, int preEnd, vector<int>& inorder, int inStart, int inEnd, unordered_map<int,int> inOrderIndicies)
{
    if (preStart > preEnd || inStart > inEnd)
        return nullptr;

    TreeNode* root = new TreeNode(preorder[preStart]);

    int mid = inOrderIndicies[preorder[preStart]];
    int leftSubTreeSize = mid - inStart;

    root->left = buildTreeHelper(preorder, preStart + 1, preStart + leftSubTreeSize, inorder, inStart, mid - 1, inOrderIndicies);
    root->right = buildTreeHelper(preorder, preStart + leftSubTreeSize + 1, preEnd, inorder, mid + 1, inEnd, inOrderIndicies);

    return root;
}

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder)
{
    unordered_map<int,int> inOrderIndicies;
    for(int i = 0; i < inorder.size(); i++)
    {
        inOrderIndicies[inorder[i]] = i;
    }

    return buildTreeHelper(preorder, 0, preorder.size()-1, inorder, 0, inorder.size()-1, inOrderIndicies);
}

/*
Problem: Binary Tree Level Order Traversal
Leet Code Link: https://leetcode.com/problems/binary-tree-level-order-traversal/description/
*/
vector<vector<int>> levelOrder(TreeNode* root)
{
    if (root == nullptr)
    {
        return {};
    }

    queue<TreeNode*> bfs;
    bfs.push(root);

    vector<vector<int>> output;
    vector<int> currentLevel;

    while (!bfs.empty())
    {
        currentLevel.clear();
        int currentLevelSize = bfs.size();
        for (int i = 0; i < currentLevelSize; i++)
        {
            TreeNode* currentNode = bfs.front();
            bfs.pop();
            currentLevel.push_back(currentNode->val);
            if (currentNode->left != nullptr)
            {
                bfs.push(currentNode->left);
            }

            if (currentNode->right != nullptr)
            {
                bfs.push(currentNode->right);
            }
        }

        output.push_back(currentLevel);
    }

    return output;
}

/*
Problem: Binary Tree Right Side View
Leet Code Link: https://leetcode.com/problems/binary-tree-right-side-view/description/
*/
vector<int> rightSideView(TreeNode* root)
{
    if (root == nullptr)
    {
        return {};
    }

    vector<int> output;
    queue<TreeNode*> bfs;
    bfs.push(root);

    TreeNode* current = nullptr;
    while (!bfs.empty())
    {
        TreeNode* rightNode = nullptr;
        int bfsQueueSize = bfs.size();
        for (int i = 0; i<bfsQueueSize; i++)
        {
            current = bfs.front();
            bfs.pop();

            if (current != nullptr)
            {
                rightNode = current;
                bfs.push(current->left);
                bfs.push(current->right);
            }
        }

        if (rightNode != nullptr)
        {
            output.push_back(rightNode->val);
        }
    }

    return output;
}

/*
Problem: Design Binary Search Tree
Leet Code Link:
*/
class TreeMap {

private:
    TreeNode2* root;

    void InOrderTraversal(vector<int>& output, TreeNode2* root)
    {
        if (root == nullptr)
            return;

        InOrderTraversal(output, root->left);
        output.push_back(root->key);
        InOrderTraversal(output, root->right);
    }

    TreeNode2* findMin(TreeNode2* node)
    {
        if (node == nullptr)
            return nullptr;

        if (node->left == nullptr)
        {
            return node;
        }

        return findMin(node->left);
    }

    TreeNode2* findMax(TreeNode2* node)
    {
        if (node == nullptr)
            return nullptr;

        if (node->right == nullptr)
        {
            return node;
        }

        return findMax(node->right);
    }


    TreeNode2* GetSelectedTreeNode(TreeNode2* root, int key)
    {
        if (root == nullptr)
            return nullptr;

        if (root->key == key)
        {
            return root;
        }
        else if (root->key > key)
        {
            return GetSelectedTreeNode(root->left, key);
        }
        else
        {
            return GetSelectedTreeNode(root->right, key);
        }
    }

    void InsertTreeNode(TreeNode2*& root, int key, int val)
    {
        if (root == nullptr)
        {
            root = new TreeNode2(key, val);
            return;
        }

        if (root->key == key)
        {
            root->value = val;
            return;
        }
        else if (root->key > key)
        {
            InsertTreeNode(root->left, key, val);
        }
        else
        {
            InsertTreeNode(root->right, key, val);
        }
    }

    TreeNode2* RemoveTreeNode(TreeNode2* curr, int key)
    {
        if (curr == nullptr)
        {
            return nullptr;
        }

        if (key > curr->key)
        {
            curr->right = RemoveTreeNode(curr->right, key);
        }
        else if (key < curr->key)
        {
            curr->left = RemoveTreeNode(curr->left, key);
        }
        else
        {
            if (curr->left == nullptr)
            {
                return curr->right;
            }
            else if (curr->right == nullptr)
            {
                return curr->left;
            }
            else
            {
                TreeNode2* minNode = findMin(curr->right);
                curr->key = minNode->key;
                curr->value = minNode->value;
                curr->right = RemoveTreeNode(curr->right, minNode->key);
            }
        }

        return curr;
    }

public:
    TreeMap() : root(nullptr) {}

    void insert(int key, int val)
    {
        InsertTreeNode(root, key, val);
    }

    int get(int key)
    {
        TreeNode2* selectedNode = GetSelectedTreeNode(root, key);
        return selectedNode != nullptr ? selectedNode->value : -1;
    }

    int getMin()
    {
        TreeNode2* minNode = findMin(root);
        return minNode != nullptr ? minNode->value : -1;
    }

    int getMax()
    {
        TreeNode2* maxNode = findMax(root);
        return maxNode != nullptr ? maxNode->value : -1;
    }

    void remove(int key)
    {
        root = RemoveTreeNode(root, key);
    }

    std::vector<int> getInorderKeys()
    {
        vector<int> output;
        InOrderTraversal(output, root);
        return output;
    }
};

/*
Problem: Path Sum
Leet Code Link: https://leetcode.com/problems/path-sum/
*/
bool hasPathSum(TreeNode* root, int targetSum)
{
    if (root == nullptr)
        return false;

    if (root->left == nullptr && root->right == nullptr)
        return root->val == targetSum;

    if (hasPathSum(root->left, targetSum - root->val))
        return true;

    else if (hasPathSum(root->right, targetSum - root->val))
        return true;

    return false;
}

/*
Problem: Subsets
Leet Code Link: https://leetcode.com/problems/subsets/description/
*/
void dfs(const vector<int>& nums, int i, vector<int>& subset, vector<vector<int>>& res)
{
    if (i >= nums.size())
    {
        res.push_back(subset);
        return;
    }

    subset.push_back(nums[i]);
    dfs(nums, i+1, subset, res);

    subset.pop_back();
    dfs(nums, i+1, subset, res);
}

vector<vector<int>> subsets(vector<int>& nums)
{
    vector<vector<int>> res;
    vector<int> subset;
    dfs(nums, 0, subset, res);
    return res;
}

/*
Problem: Kth Largest Element in a Stream
Leet Code Link: https://leetcode.com/problems/kth-largest-element-in-a-stream/description/
*/
class KthLargest {
public:
    int kth;
    priority_queue<int, vector<int>, greater<int>> heap;

    KthLargest(int k, vector<int>& nums) : kth(k)
    {
        for (int num : nums)
        {
            heap.push(num);
            if (heap.size() > k)
            {
                heap.pop();
            }
        }
    }

    int add(int val)
    {
        heap.push(val);
        if (heap.size() > kth)
        {
            heap.pop();
        }

        return heap.top();
    }
};

/*
Problem: Last Stone Weight
Leet Code Link: https://leetcode.com/problems/last-stone-weight/description/
*/
int lastStoneWeight(vector<int>& stones)
{
    priority_queue<int, vector<int>, less<int>> heap;
    for (int stone : stones)
    {
        heap.push(stone);
    }

    while (heap.size() >= 2)
    {
        int first = heap.top();
        heap.pop();
        int second = heap.top();
        heap.pop();

        if (first > second)
        {
            heap.push(first - second);
        }
    }

    heap.push(0);
    return heap.top();
}

/*
Problem: K Closest Points to Origin
Leet Code Link: https://leetcode.com/problems/k-closest-points-to-origin/
*/
vector<vector<int>> kClosest(vector<vector<int>>& points, int k)
{
    priority_queue<pair<int, vector<int>>> maxHeap;
    for (auto& point : points)
    {
        int distance = point[0]*point[0] + point[1]*point[1];
        pair<int, vector<int>> distanceWithPoint = {distance, point};
        maxHeap.push(distanceWithPoint);

        if (maxHeap.size() > k)
        {
            maxHeap.pop();
        }
    }

    vector<vector<int>> res;
    while (!maxHeap.empty())
    {
        pair<int, vector<int>> distanceWithPoint = maxHeap.top();
        maxHeap.pop();
        res.push_back(distanceWithPoint.second);
    }
    return res;
}

/*
Problem: Design Heap
Leet Code Link:
*/
class MinHeap {
private:
    vector<int> heap;

    void BubbleUp(int index)
    {
        if (heap.size() <= 1)
        {
            return;
        }


        while (index > 1)
        {
            int child = index;
            int parent = child / 2;
            if (heap[child] < heap[parent])
            {
                swap(heap[child], heap[parent]);
                child = parent;
                index = parent;
            }
            else
                break;
        }
    }

    void BubbleDown(int index)
    {
        int size = heap.size();
        int child = 2 * index;

        while (child < size)
        {
            if (child + 1 < size && heap[child + 1] < heap[child] && heap[child + 1] < heap[index])
            {
                swap(heap[child + 1], heap[index]);
                index = child + 1;
            }
            else if (heap[child] < heap[index])
            {
                swap(heap[child], heap[index]);
                index = child;
            }
            else
                break;

            child = 2 * index;
        }
    }

public:
    MinHeap()
    {
        heap.push_back(0);  // Reserve index 0, heap starts at index 1
    }

    void push(int val)
    {
        heap.push_back(val);
        BubbleUp(heap.size() - 1);
    }

    int pop()
    {
        if (heap.size() <= 1)
        {
            return -1;  // No element to pop
        }

        int pop_val = heap[1];  // Store the top element to return it later

        if (heap.size() == 2)
        {
            heap.pop_back();  // Only one element left, just remove it
            return pop_val;
        }

        // Replace the root element with the last element
        heap[1] = heap.back();
        heap.pop_back();

        // Restore the heap property by bubbling down the new root element
        BubbleDown(1);

        return pop_val;  // Return the popped element
    }

    int top()
    {
        if (heap.size() <= 1)
        {
            return -1;  // No top element
        }

        return heap[1];  // Return the root element
    }

    void heapify(const vector<int>& arr)
    {
        heap.clear();
        heap.push_back(0);
        heap.insert(heap.end(), arr.begin(), arr.end());

        int midPoint = (heap.size() - 1) / 2;
        while (midPoint > 0)
        {
            BubbleDown(midPoint);
            midPoint--;
        }
    }
};

/*
Problem: Kth Largest Element in an Array
Leet Code Link: https://leetcode.com/problems/kth-largest-element-in-an-array/description/
*/
int QuickSelectFindKthLargest(vector<int>& nums, int k, int left, int right)
{
    int pivot = nums[right];
    int p = left;

    for (int i = left; i < right; i++)
    {
        if (nums[i] <= pivot)
        {
            swap(nums[i] , nums[p]);
            p++;
        }
    }

    swap(nums[right] , nums[p]);

    if (p > k)
    {
        return QuickSelectFindKthLargest(nums, k, left, p - 1);
    }
    else if (p < k)
    {
        return QuickSelectFindKthLargest(nums, k, p + 1, right);
    }
    else
    {
        return nums[p];
    }
}

int findKthLargest(vector<int>& nums, int k)
{
    int numsSize = nums.size();
    return QuickSelectFindKthLargest(nums, numsSize - k, 0, numsSize - 1);
}

/*
Problem: LRU Cache
Leet Code Link: https://leetcode.com/problems/lru-cache/description/
*/
class LRUCache
{
private:
    unordered_map<int, pair<int, list<int>::iterator>> cache;
    list<int> order;
    int capacity;

public:
    LRUCache(int capacity)
    {
        this->capacity = capacity;
    }

    int get(int key)
    {
        if (cache.find(key) == cache.end())
        {
            return -1;
        }

        order.erase(cache[key].second);
        order.push_back(key);
        cache[key].second = --order.end();

        return cache[key].first;
    }

    void put(int key, int value)
    {
        if (cache.find(key) != cache.end())
        {
            order.erase(cache[key].second);
        }
        else if (cache.size() == capacity)
        {
            int lru = order.front();
            order.pop_front();
            cache.erase(lru);
        }

        order.push_back(key);
        cache[key] = {value, --order.end()};
    }
};

/*
Problem: Design Hash Table
Leet Code Link:
*/
class HashTable
{
private:
    int size;
    int capacity;
    vector<vector<pair<int,int>>> hashTable;


    int HashFunction(int key)
    {
        return key % capacity;
    }

    int HashFunction(int key, int customCapacity)
    {
        return key % customCapacity;
    }

public:
    HashTable(int capacityOfHashTable) : size(0), capacity(capacityOfHashTable)
    {
        hashTable = vector<vector<pair<int,int>>>(capacity);
    }

    void insert(int key, int value)
    {
        int hash = HashFunction(key);

        for(pair<int,int>& keyValue : hashTable[hash])
        {
            if (keyValue.first == key)
            {
                keyValue.second = value;
                return;
            }
        }

        hashTable[hash].push_back(pair<int,int>(key,value));
        size++;

        if ((float)size / capacity >= 0.5)
        {
            resize();
        }
    }

    int get(int key)
    {
        int hash = HashFunction(key);

        for(pair<int,int>& keyValue : hashTable[hash])
        {
            if (keyValue.first == key)
            {
                return keyValue.second;
            }
        }

        return -1;
    }

    bool remove(int key)
    {
        int hash = HashFunction(key);
        vector<pair<int,int>>& bucket = hashTable[hash];

        for(int i = 0; i < bucket.size(); i++)
        {
            if (bucket[i].first == key)
            {
                bucket.erase(bucket.begin() + i);
                size--;
                return true;
            }
        }

        return false;
    }

    int getSize() const
    {
        return size;
    }

    int getCapacity() const
    {
        return capacity;
    }

    void resize()
    {
        int newCapacitySize = capacity * 2;
        vector<vector<pair<int,int>>> newHashTable(newCapacitySize);

        for (vector<pair<int,int>>& bucket : hashTable)
        {
            for (pair<int,int>& keyValuePair : bucket)
            {
                int hash = HashFunction(keyValuePair.first, newCapacitySize);
                newHashTable[hash].push_back(keyValuePair);
            }
        }

        hashTable = newHashTable;
        capacity = newCapacitySize;
    }
};

/*
Problem: Matrix Depth-First Search
Leet Code Link:
*/
int countPaths(vector<vector<int>>& grid, unordered_set<string>& visited, int row, int columb)
{
    int ROW = grid.size();
    int COL = grid[0].size();
    int count = 0;

    if (min(row, columb) < 0 || row == ROW || columb == COL || grid[row][columb] == 1 || visited.find(to_string(row) + "," + to_string(columb)) != visited.end())
    {
        return 0;
    }
    else if (row == ROW - 1 && columb == COL - 1 )
    {
        return 1;
    }

    visited.insert(to_string(row) + "," + to_string(columb));

    count += countPaths(grid, visited, row + 1, columb);
    count += countPaths(grid, visited, row - 1, columb);
    count += countPaths(grid, visited, row, columb + 1);
    count += countPaths(grid, visited, row, columb - 1);

    visited.erase(to_string(row) + "," + to_string(columb));

    return count;
}

int countPaths(vector<vector<int>>& grid)
{
    unordered_set<string> visited;
    return countPaths(grid, visited, 0, 0);
}

/*
Problem: Number of Islands
Leet Code Link: https://leetcode.com/problems/number-of-islands/description/
*/
void dfsNumIsIsland(vector<vector<char>>& grid, int r, int c)
{
    if (min(r,c) < 0 || r >= grid.size() || c >= grid[0].size() || grid[r][c] == '0')
        return;

    grid[r][c] = '0';
    dfsNumIsIsland(grid, r + 1, c);
    dfsNumIsIsland(grid, r - 1, c);
    dfsNumIsIsland(grid, r, c + 1);
    dfsNumIsIsland(grid, r, c -1);
}

int numIslands(vector<vector<char>>& grid)
{
    int numberOfIslands = 0;
    int ROW = grid.size(), COL = grid[0].size();

    for (int r = 0; r < ROW; r++)
    {
        for (int c = 0; c < COL; c++)
        {
            if (grid[r][c] == '1')
            {
                dfsNumIsIsland(grid, r, c);
                numberOfIslands++;
            }
        }
    }

    return numberOfIslands;
}

/*
Problem: Max Area of Island
Leet Code Link: https://leetcode.com/problems/max-area-of-island/description/
*/
int dfsMaxAreaIsland(vector<vector<int>>& grid, int r, int c)
{
    if (min(r,c) < 0 || r >= grid.size() || c >= grid[0].size() || grid[r][c] == 0)
        return 0;

    int islandMax = 1;
    grid[r][c] = 0;

    islandMax += dfsMaxAreaIsland(grid, r + 1, c);
    islandMax += dfsMaxAreaIsland(grid, r - 1, c);
    islandMax += dfsMaxAreaIsland(grid, r, c + 1);
    islandMax += dfsMaxAreaIsland(grid, r, c - 1);

    return islandMax;
}

int maxAreaOfIsland(vector<vector<int>>& grid)
{
    int maxIslanArea = 0;
    int ROW = grid.size(), COL = grid[0].size();

    for (int r = 0; r < ROW; r++)
    {
        for (int c = 0; c < COL; c++)
        {
            if (grid[r][c] == 1)
            {
                int island = dfsMaxAreaIsland(grid, r, c);
                maxIslanArea = max(maxIslanArea, island);
            }
        }
    }

    return maxIslanArea;
}

/*
Problem: Matrix Breadth-First Search
Leet Code Link:
*/
int shortestPath(vector<vector<int>>& grid)
{
    int ROW = grid.size(), COL = grid[0].size();
    unordered_set<string> visit;
    queue<pair<int, int>> queue;
    queue.push({0, 0});
    visit.insert("0,0");

    int shortestLength = 0;
    while (!queue.empty())
    {
        int levelSize = queue.size();
        for(int i=0; i< levelSize; i++)
        {
            pair<int,int> currentCell = queue.front();
            queue.pop();

            if (currentCell.first == ROW - 1 && currentCell.second == COL - 1)
            {
                return shortestLength;
            }

            vector<pair<int,int>> neigbours = {{1,0},{-1,0},{0,1},{0,-1}};
            for (pair<int,int>& neigbour : neigbours)
            {
                int r = currentCell.first + neigbour.first;
                int c = currentCell.second + neigbour.second;
                string cell = to_string(r) + "," + to_string(c);
                if(min(r,c)< 0 || r >= ROW || c >= COL || grid[r][c] == 1 || visit.find(cell) != visit.end())
                    continue;

                queue.push({r,c});
                visit.insert(cell);
            }
        }

        shortestLength++;
    }

    return -1;
}

/*
Problem: Shortest Path in Binary Matrix
Leet Code Link: https://leetcode.com/problems/shortest-path-in-binary-matrix/description/
*/
int shortestPathBinaryMatrix(vector<vector<int>>& grid)
{
    int ROW = grid.size(), COL = grid[0].size();
    unordered_set<string> visit;
    queue<pair<int, int>> bfsQueue;

    if (grid[0][0] == 1 || grid[ROW - 1][COL - 1] == 1) {
        return -1;
    }

    bfsQueue.push({0, 0});
    visit.insert("0,0");

    int shortestLength = 0;
    while (!bfsQueue.empty())
    {
        int bfsLevelSize = bfsQueue.size();
        for (int i=0; i < bfsLevelSize; i++)
        {
            pair<int,int> currentCell = bfsQueue.front();
            bfsQueue.pop();
            int R = currentCell.first;
            int C = currentCell.second;

            if (R == ROW - 1 && C == COL - 1)
            {
                return shortestLength + 1;
            }

            vector<pair<int,int>> neigbours = {{0,1},{0,-1},{1,0},{-1,0},{1,1},{1,-1},{-1,1},{-1,-1}};
            for (pair<int,int> neigbour : neigbours)
            {
                int r = R + neigbour.first;
                int c = C + neigbour.second;
                string visitedCell = to_string(r)+","+to_string(c);
                if (min(r,c) < 0 || r >= ROW || c >= COL || visit.find(visitedCell) != visit.end() || grid[r][c] == 1)
                {
                    continue;
                }

                bfsQueue.push({r, c});
                visit.insert(visitedCell);
            }
        }
        shortestLength++;
    }

    return -1;
}

/*
Problem: Rotting Fruit
Leet Code Link: https://leetcode.com/problems/rotting-oranges/description/
*/
int orangesRotting(vector<vector<int>>& grid)
{
    int ROW = grid.size(), COL = grid[0].size();
    int ripeFruitCount = 0;
    int minutes = 0;
    queue<pair<int,int>> bfsQueue;

    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            if(grid[i][j] == 2)
            {
                bfsQueue.push({i,j});
            }

            if(grid[i][j] == 1)
            {
                ripeFruitCount++;
            }
        }
    }

    if (ripeFruitCount == 0)
    {
        return 0;
    }

    while (!bfsQueue.empty())
    {
        int bfsLevelSize = bfsQueue.size();
        bool rottenAtBFSLevel = false;
        for (int i = 0; i < bfsLevelSize; i++)
        {
            pair<int,int> cCell = bfsQueue.front();
            bfsQueue.pop();

            int R = cCell.first;
            int C = cCell.second;

            vector<pair<int,int>> neigbours = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
            for (pair<int,int> neigbour : neigbours)
            {
                int r = R + neigbour.first;
                int c = C + neigbour.second;
                if (r < 0 || r >= ROW || c < 0 || c >= COL || grid[r][c] != 1)
                    continue;

                ripeFruitCount--;
                bfsQueue.push({r,c});
                grid[r][c] = 2;
                rottenAtBFSLevel = true;
            }


        }

        if (rottenAtBFSLevel)
        {
            minutes++;
        }
    }

    return ripeFruitCount == 0 ? minutes : -1;
}

/*
Problem: Design Graph
Leet Code Link:
*/
class Graph
{
private:
    unordered_map<int, unordered_set<int>> adj_list;

    bool dfsHelper(int src, int dst, unordered_set<int>& visited)
    {
        if (src == dst)
            return true;

        for (const int &neighbor : adj_list[src])
        {
            if (visited.find(neighbor) == visited.end())
            {
                if (dfsHelper(neighbor, dst, visited))
                    return true;
            }
        }

        return false;
    }

public:
    Graph() {}

    void addEdge(int src, int dst)
    {
        adj_list[src].insert(dst);
    }

    bool removeEdge(int src, int dst)
    {
        if (adj_list.find(src) == adj_list.end() || adj_list[src].find(dst) == adj_list[src].end()) {
            return false;
        }

        adj_list[src].erase(dst);
    }

    bool hasPath(int src, int dst)
    {
        unordered_set<int> visited;
        return dfsHelper(src, dst, visited);
    }
};

/*
Problem: Clone Graph
Leet Code Link: https://leetcode.com/problems/clone-graph/description/
*/
Node* DFSCloneGraph(Node* node, map<Node*, Node*>& oldToNew)
{
    if (node == nullptr)
        return nullptr;

    if (oldToNew.count(node))
    {
        return oldToNew[node];
    }

    Node* copy = new Node(node->val);
    oldToNew[node] = copy;

    for (Node* adjNode : node->neighbors)
    {
        copy->neighbors.push_back(DFSCloneGraph(adjNode, oldToNew));
    }

    return copy;
}

Node* cloneGraph(Node* node)
{
    map<Node*, Node*> oldToNew;
    return DFSCloneGraph(node, oldToNew);
}

/*
Problem: Course Schedule
Leet Code Link: https://leetcode.com/problems/course-schedule/
*/
bool DFSCanFinish(int curr, unordered_map<int, vector<int>>& preMap, unordered_set<int>& visited)
{
    if (visited.count(curr))
        return false;

    if (preMap[curr].empty())
        return true;

    visited.insert(curr);
    for (int adj : preMap[curr])
    {
        if (!DFSCanFinish(adj, preMap, visited))
            return false;
    }
    visited.erase(curr);
    preMap[curr].clear();

    return true;
}

bool canFinish(int numCourses, vector<vector<int>>& prerequisites)
{
    unordered_map<int, vector<int>> preMap;
    unordered_set<int> visited;

    for (const auto& prereq : prerequisites)
    {
        preMap[prereq[0]].push_back(prereq[1]);
    }

    for (int c = 0; c < numCourses; c++)
    {
        if (!DFSCanFinish(c, preMap, visited))
            return false;
    }

    return true;
}

/*
Problem: Climbing Stairs
Leet Code Link: https://leetcode.com/problems/climbing-stairs/description/
*/
int climbStairs(int n)
{
    int one = 1;
    int two = 2;

    if (n == 1) {
        return one;
    }
    if (n == 2) {
        return two;
    }

    int result = 0;
    int i = 3;
    while (i <= n)
    {
        result = one + two;
        one = two;
        two = result;
        i++;
    }

    return result;
}

/*
Problem: House Robber
Leet Code Link: https://leetcode.com/problems/house-robber/
*/
int rob(vector<int>& nums)
{
    if (nums.size() == 1)
    {
        return nums[0];
    }

    int first = 0;
    int second = 0;

    for (int num : nums)
    {
        int temp = max(num + first, second);
        first = second;
        second = temp;
    }

    return second;
}

/*
Problem: Unique Paths
Leet Code Link: https://leetcode.com/problems/unique-paths/description/
*/
int topdownmemoization(int i, int j, int m, int n, vector<vector<int>>& memo)
{
    if (i >= m || j >= n)
        return 0;

    if (i == m - 1 || j == n - 1)
        return 1;

    if (memo[i][j] != -1)
        return memo[i][j];

    memo[i][j] = topdownmemoization(i + 1, j, m, n, memo) + topdownmemoization(i, j + 1, m, n, memo);

    return memo[i][j];
}

//Top-Down approach with momoization
int uniquePathsTopDown(int m, int n)
{
    vector<vector<int>>memo(m, vector<int>(n, -1));
    return topdownmemoization(0, 0, m, n, memo);
}

//Bottom-Up approach
int uniquePathsBottomUp(int m, int n)
{
    vector<int> row(n,0);
    for (int i = m - 1; i >= 0; i--)
    {
        vector<int> currentRow(n,1);
        for (int j = n - 2; j >= 0; j--)
        {
            currentRow[j] = currentRow[j+1] + row[j];
        }

        row = currentRow;
    }

    return row[0];
}

/*
Problem: Unique Paths 2
Leet Code Link: https://leetcode.com/problems/unique-paths-ii/
*/
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    const int ROW = obstacleGrid.size();
    const int COL = obstacleGrid[0].size();
    vector<long> dp(COL);
    dp[COL - 1] = 1;

    for (int r = ROW - 1; r >= 0; --r)
    {
        for (int c = COL - 1; c >= 0; --c)
        {
            if (obstacleGrid[r][c] == 1)
            {
                dp[c] = 0;
            }

            else if (c + 1 < COL)
            {
                dp[c] += dp[c + 1];
            }
        }
    }
    return dp[0];
}

/*
Problem: Longest Common Subsequence
Leet Code Link: https://leetcode.com/problems/longest-common-subsequence/
*/
int longestCommonSubsequence(string text1, string text2)
{
    vector<vector<int>> dp(text1.size() + 1, vector<int>(text2.size() + 1));

    for (int i = text1.size() - 1; i >= 0; i--)
    {
        for (int j = text2.size() - 1; j >= 0; j--)
        {
            if (text1[i] == text2[j])
            {
                dp[i][j] = 1 + dp[i + 1][j + 1];
            }
            else
            {
                dp[i][j] = max(dp[i][j + 1], dp[i + 1][j]);
            }
        }
    }

    return dp[0][0];
}

/*
Problem: Number Of 1 Bits
Leet Code Link: https://leetcode.com/problems/number-of-1-bits/
*/
int hammingWeight(uint32_t n)
{
    int count = 0;
    while (n > 0)
    {
        if (n & 1 == 1)
        {
            count++;
        }

        n = n >> 1;
    }

    return count;
}

/*
Problem: Counting Bits
Leet Code Link: https://leetcode.com/problems/counting-bits/description/
*/
vector<int> countBits(int n)
{
    vector<int> dp(n + 1);
    for (int i = 1; i <= n; i++)
    {
        dp[i] = dp[i >> 1] + (i & 1);
    }

    return dp;
}

/*
Problem: Resverse Bits
Leet Code Link: https://leetcode.com/problems/reverse-bits/
*/
uint32_t reverseBits(uint32_t n)
{
    uint32_t res = 0;
    for (int i = 0; i < 32; i++)
    {
        uint32_t bit = (n >> i) & 1;
        res |= (bit << (31 - i));
    }

    return res;
}

/*
Problem: Resverse Maximum Subarray
Leet Code Link: https://leetcode.com/problems/maximum-subarray/
*/
int maxSubArray(vector<int>& nums)
{
    if (nums.size() == 0)
        return 0;

    int maxSubArrayLength = nums[0] , currentMaxAraryLength = 0;

    for (int& num : nums)
    {
        currentMaxAraryLength = max(currentMaxAraryLength +  num, num);

        maxSubArrayLength = max(maxSubArrayLength,currentMaxAraryLength);
    }

    return maxSubArrayLength;
}

/*
Problem: Maximum Sum Circular Subarray
Leet Code Link: https://leetcode.com/problems/maximum-sum-circular-subarray/
*/
int maxSubarraySumCircular(vector<int>& nums)
{
    int maxSum = nums[0], minSum = nums[0], curMax=0, curMin=0;
    int totalSum = 0;

    for(int num : nums)
    {
        curMax = max(curMax + num, num);
        maxSum = max(maxSum, curMax);
        curMin = min(curMin + num, num);
        minSum = min(minSum, curMin);
        totalSum+=num;
    }

    return maxSum>0? max(maxSum, totalSum-minSum) : maxSum;
}

/*
Problem: Longest Turbulent Subarray
Leet Code Link: https://leetcode.com/problems/longest-turbulent-subarray/
*/
int maxTurbulenceSize(vector<int>& arr)
{
    int n = arr.size();
    if (n <= 1)
    {
        return n;
    }

    int pos = 1;
    int neg = 1;
    int maxLen = 1;

    for (int i = 1; i < n; ++i)
    {
        if (arr[i] > arr[i - 1])
        {
            pos = neg + 1;
            neg = 1;
        }
        else if (arr[i] < arr[i - 1])
        {
            neg = pos + 1;
            pos = 1;
        }
        else
        {
            pos = 1;
            neg = 1;
        }

        maxLen = max(maxLen, max(pos, neg));
    }

    return maxLen;
}

/*
Problem: Contains Duplicate II
Leet Code Link: https://leetcode.com/problems/contains-duplicate-ii/
*/
bool containsNearbyDuplicate(vector<int>& nums, int k)
{
    unordered_set<int> window;
    int left = 0;

    for (int right = 0; right < nums.size(); right++)
    {
        if (right - left > k)
        {
            window.erase(nums[left]);
            left++;
        }

        if (window.count(nums[right]))
            return true;

        window.insert(nums[right]);
    }

    return false;
}

/*
Problem: Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold
Leet Code Link: https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/description/
*/
int numOfSubarrays(vector<int>& arr, int k, int threshold)
{
    vector<int> window;
    int total = 0;

    for (int right = 0; right < arr.size(); right++)
    {
        window.push_back(arr[right]);

        if (window.size() > k)
        {
            window.erase(window.begin());
        }

        if ((window.size() == k) && (accumulate(window.begin(), window.end(), 0) / k >= threshold))
        {
            total++;
        }
    }

    return total;
}

/*
Problem: Minimum Size Subarray Sum
Leet Code Link: https://leetcode.com/problems/minimum-size-subarray-sum/
*/
int minSubArrayLen(int target, vector<int>& nums)
{
    int max = nums.size()+1;
    int total = 0;
    int left = 0;

    for (int right = 0; right < nums.size(); right++)
    {
        total += nums[right];
        while (total >= target)
        {
            max = min(right - left + 1, max);
            total -= nums[left];
            left++;
        }
    }

    return (max == nums.size() + 1 ? 0 : max);
}

/*
Problem: Longest Substring Without Repeating Characters
Leet Code Link: https://leetcode.com/problems/longest-substring-without-repeating-characters/
*/
int lengthOfLongestSubstring(string s)
{
    int sSize = s.size();
    unordered_map<char, int> windowMap;
    int left = 0;
    int result = 0;

    for (int right = 0; right < sSize; right++)
    {
        if (windowMap.find(s[right]) != windowMap.end())
        {
            left = max(left, windowMap[s[right]] + 1);
        }

        windowMap[s[right]] = right;
        result = max(result, right - left + 1);
    }

    return result;
}

/*
Problem: Longest Repeating Character Replacement
Leet Code Link: https://leetcode.com/problems/longest-repeating-character-replacement/
*/
int characterReplacement(std::string s, int k)
    {
        unordered_map<char, int> frequencyCount;
        int result = 0;
        int left = 0;
        int maxFrequency = 0;

        for (int right = 0; right < s.size(); right++)
        {
            frequencyCount[s[right]]++;
            maxFrequency = max(maxFrequency, frequencyCount[s[right]]);

            if ((right - left + 1) - maxFrequency > k)
            {
                frequencyCount[s[left]]--;
                left++;
            }

            result = max(result, (right - left + 1));
        }

        return result;
    }

/*
Problem: Range Sum Query - Immutable
Leet Code Link: Problem: https://leetcode.com/problems/range-sum-query-immutable/
*/
class NumArray {
    public:
        vector<int> sum;

        NumArray(vector<int>& nums) {
            sum.resize(nums.size() + 1, 0);
            for (int i = 0; i < nums.size(); i++) {
                sum[i + 1] = sum[i] + nums[i];
            }
        }

        int sumRange(int left, int right) {
            return sum[right + 1] - sum[left];
        }
    };

/*
Problem: Range Sum Query 2D - Immutable
Leet Code Link: Problem: https://leetcode.com/problems/range-sum-query-2d-immutable/
*/
class NumMatrix {
    public:
        vector<vector<int>> prefixMatrix;

        NumMatrix(vector<vector<int>>& matrix)
        {
            if (matrix.size() == 0 || matrix[0].size() == 0)
                return;

            prefixMatrix.resize(matrix.size(), vector<int>(matrix[0].size() + 1,0));
            for (int r = 0; r < matrix.size(); r++)
            {
                for (int c = 0; c < matrix[0].size(); c++)
                {
                    prefixMatrix[r][c + 1] = prefixMatrix[r][c] + matrix[r][c];
                }
            }
        }

        int sumRegion(int row1, int col1, int row2, int col2)
        {
            int sum = 0;
            for (int row = row1; row <= row2; row++)
            {
                sum += prefixMatrix[row][col2+1] - prefixMatrix[row][col1];
            }

            return sum;
        }
    };

/*
Problem: Find Pivot Index
Leet Code Link: Problem: https://leetcode.com/problems/find-pivot-index/description/
*/
int pivotIndex(vector<int>& nums)
{
    int numsSize = nums.size();
    vector<int> prefixSum(numsSize + 1, 0);

    for (int i = 0; i< numsSize; i++)
    {
        prefixSum[i + 1] = prefixSum[i] + nums[i];
    }

    int sizePrefix = prefixSum.size();
    for (int p = 1; p < sizePrefix; p++)
    {
        if (prefixSum[p-1] == (prefixSum[sizePrefix - 1] - prefixSum[p]))
        {
            return p - 1;
        }
    }

    return -1;
}

/*
Problem: Subarray Sum Equals K
Leet Code Link: Problem: https://leetcode.com/problems/subarray-sum-equals-k/
*/
int subarraySum(vector<int>& nums, int k)
{
    int numsSize = nums.size();
    int sum = 0;
    int count = 0;
    unordered_map<int,int> sums;

    for (int p = 0; p < numsSize; p++)
    {
        sum += nums[p];
        if (sum == k)
        {
            count++;
        }

        if (sums.find(sum - k) != sums.end())
        {
            count += sums[sum - k];
        }

        sums[sum]++;
    }

    return count;
}

/*
Problem: Middle of the Linked List
Leet Code Link: Problem: https://leetcode.com/problems/middle-of-the-linked-list/
*/
ListNode* middleNode(ListNode* head)
{
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast != nullptr && fast->next != nullptr)
    {
        slow = slow->next;
        fast = fast->next->next;
    }

    return slow;
}

/*
Problem: Maximum Twin Sum of a Linked List
Leet Code Link: Problem: https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/
*/
int pairSum(ListNode* head)
{
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast != nullptr && fast->next != nullptr)
    {
        slow = slow->next;
        fast = fast->next->next;
    }

    ListNode* nextNode;
    ListNode* prev = nullptr;
    while (slow != nullptr)
    {
        nextNode = slow->next;
        slow->next = prev;
        prev = slow;
        slow = nextNode;
    }

    int maximumSum = 0;
    ListNode* start = head;
    while (prev != nullptr)
    {
        maximumSum = max(maximumSum, start->val + prev->val);
        prev = prev->next;
        start = start->next;
    }

    return maximumSum;
}

/*
Problem: Linked List Cycle II
Leet Code Link: Problem: https://leetcode.com/problems/linked-list-cycle-ii/description/
*/
ListNode *detectCycle(ListNode *head)
{
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast != nullptr && fast->next != nullptr)
    {
        slow = slow->next;
        fast = fast->next->next;

        if (slow == fast) {
            break;
        }
    }

    if (fast == nullptr || fast->next == nullptr)
        return nullptr;

    fast = head;
    while (fast != slow)
    {
        slow = slow->next;
        fast = fast->next;
    }

    return fast;
}

/*
Problem: Find the Duplicate Number
Leet Code Link: https://leetcode.com/problems/find-the-duplicate-number/description/
*/
int findDuplicate(vector<int>& nums)
{
    if (nums.size() == 1)
    {
        return nums[0];
    }

    int slow = 0;
    int fast = 0;

    while (true)
    {
        slow = nums[slow];
        fast = nums[nums[fast]];

        if (fast == slow)
            break;
    }

    fast = 0;
    while (fast != slow)
    {
        slow = nums[slow];
        fast = nums[fast];
    }

    return fast;
}


/*
Problem: Implement Trie (Prefix Tree)
Leet Code Link: https://leetcode.com/problems/implement-trie-prefix-tree/description/
*/
class Trie {
public:

    TrieNode* root;

    Trie(): root(new TrieNode()) {}

    void insert(string word)
    {
        TrieNode* current = root;
        for (char c : word)
        {
            if (current->children.count(c) == 0)
            {
                current->children[c] = new TrieNode();
            }

            current = current->children[c];
        }

        current->word = true;
    }

    bool search(string word)
    {
        TrieNode* current = root;
        for (char c : word)
        {
            if (current->children.count(c) == 0)
            {
                return false;
            }

            current = current->children[c];
        }

        return current->word;
    }

    bool startsWith(string prefix)
    {
        TrieNode* current = root;
        for (char c : prefix)
        {
            if (current->children.count(c) == 0)
            {
                return false;
            }

            current = current->children[c];
        }

        return true;
    }
};

/*
Problem: Design Add and Search Words Data Structure
Leet Code Link: https://leetcode.com/problems/design-add-and-search-words-data-structure/description/
*/
class WordDictionary {
public:

    TrieNode2* root;

    WordDictionary(): root(new TrieNode2()) {

    }

    void addWord(string word)
    {
        TrieNode2* curr = root;
        for (char c : word)
        {
            if (curr->children[c - 'a'] == nullptr)
            {
                curr->children[c - 'a'] = new TrieNode2();
            }

            curr = curr->children[c - 'a'];
        }

        curr->word = true;
    }

    bool search(string word)
    {
        return dfsSearch(word, 0, root);
    }

    bool dfsSearch(string word, int j, TrieNode2* root)
    {
        int wordSize = word.size();
        TrieNode2* current = root;

        for (int i = j; i < wordSize; i++)
        {
            char chr = word[i];
            if (chr == '.')
            {
                for (TrieNode2* nextChar : current->children)
                {
                    if (nextChar != nullptr && dfsSearch(word, i + 1, nextChar))
                        return true;
                }

                return false;
            }
            else
            {
                if (current->children[chr - 'a'] == nullptr)
                    return false;

                current = current->children[chr - 'a'];
            }
        }

        return current->word;
    }
};

/*
Problem: Word Search II
Leet Code Link: https://leetcode.com/problems/word-search-ii/description/
*/
class Solution
{
    public:
        int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        void addWord(TrieNode* current, string word)
        {
            for (auto ch : word)
            {
                if (current->children[ch - 'a'] == nullptr)
                {
                    current->children[ch - 'a'] = new TrieNode();
                }

                current = current->children[ch - 'a'];
            }

            current->word = true;
        }

        void dfs(vector<string>& answer, string valid, TrieNode* current, int i, int j,vector<vector<char>>& board,vector<vector<bool>>& visited)
        {
            if (current == nullptr)
                return;

            if (current->word){
                answer.push_back(valid);
                current->word = false;
            }

            visited[i][j] = true;

            for (int dir = 0; dir < 4; dir++)
            {
                int upX = i + directions[dir][0];
                int upY = j + directions[dir][1];

                if (upX >= 0 && upX < board.size() && upY >= 0 && upY < board[0].size() && !visited[upX][upY] && current->children[board[upX][upY] - 'a'] != nullptr)
                {
                    dfs(answer, valid + board[upX][upY], current->children[board[upX][upY] - 'a'], upX, upY, board, visited);
                }
            }

            visited[i][j] = false;
        }

        vector<string> findWords(vector<vector<char>>& board, vector<string>& words)
        {
            TrieNode* trieWords = new TrieNode();
            int n = board.size(), m = board[0].size();
            vector<vector<bool>> visited(n,vector<bool>(m,false));

            for (auto word : words)
            {
                addWord(trieWords, word);
            }

            vector<string> answer;
            for (int i = 0; i < n; i++) {
                for (auto j = 0; j < m; j++) {
                    char character = board[i][j];
                    if (trieWords->children[character-'a'] != nullptr)
                    {
                        dfs(answer, string(1, character), trieWords->children[character-'a'], i, j, board, visited);
                    }
                }
            }

            return answer;
        }
};

/*
Problem: Prefix and Suffix Search
Leet Code Link: https://leetcode.com/problems/prefix-and-suffix-search/description/
*/
struct TrieWF
{
    int pos = -1, maxPos = -1;
    string fullWord = "";
    TrieWF* children[26];
};

class WordFilter
{
    public:
    TrieWF* root = new TrieWF(), *revRoot = new TrieWF(), *curr;
        int res;

        void dfs(TrieWF* root, string suffix)
        {
            if (root == nullptr || root->maxPos < res)
                return;

            if (root->pos != -1 && 0 == root->fullWord.compare(root->fullWord.size() - suffix.size(), suffix.size(), suffix))
            {
                res = max(res, root->pos);
            }

            for (auto child: root->children)
            {
                if (child)
                {
                    dfs(child, suffix);
                }
            }
        }

        void populateTrie(string word, TrieWF* root, int i) {
            curr = root;
            for (char c: word)
            {
                if (!curr->children[c - 'a'])
                {
                    curr->children[c - 'a'] = new TrieWF();
                }

                curr = curr->children[c - 'a'];
                curr->maxPos = max(curr->maxPos, i);
            }
            curr->pos = i;
            curr->fullWord = word;
        }

        WordFilter(vector<string>& words)
        {
            for (int i = 0, lmt = words.size(); i < lmt; i++)
            {
                string word = words[i];
                populateTrie(word, root, i);
                reverse(begin(word), end(word));
                populateTrie(word, revRoot, i);
            }
        }

        int f(string prefix, string suffix) {
            if (prefix.size() < suffix.size())
            {
                curr = revRoot;
                reverse(prefix.begin(), prefix.end());
                reverse(suffix.begin(), suffix.end());
                swap(prefix, suffix);
            }
            else
            {
                curr = root;
            }

            res = -1;
            for (char c: prefix)
            {
                if (curr == nullptr)
                {
                    return res;
                }

                curr = curr->children[c - 'a'];
            }

            dfs(curr, suffix);
            return res;
        }
};

/*
Problem: Redundant Connection
Leet Code Link: https://leetcode.com/problems/redundant-connection/description/
*/
class DSU
{
    public:
        unordered_map<int, int> parent;
        unordered_map<int, int> rank;

        DSU(int n)
        {
            for (int i = 1; i <= n; i++)
            {
                parent[i] = i;
                rank[i] = 0;
            }
        }

        int DSUfind(int n)
        {
            int p = parent[n];
            while (p != parent[p]) {
                parent[p] = parent[parent[p]];
                p = parent[p];
            }
            return p;
        }

        bool DSUunion(int one, int two)
        {
            int first = DSUfind(one);
            int second = DSUfind(two);

            if (first == second)
                return false;

            if (rank[first] > rank[second])
            {
                parent[second] = first;
            }
            else if (rank[first] < rank[second])
            {
                parent[first] = second;
            }
            else
            {
                parent[first] = second;
                rank[second] += 1;
            }

            return true;
        }
};

class Redundant
{
    public:
        vector<int> findRedundantConnection(vector<vector<int>>& edges)
        {
            int n = edges.size();
            DSU dsu(n);

            for (auto edge : edges)
            {
                if (!dsu.DSUunion(edge[0] - 1, edge[1] - 1))
                {
                    return edge;
                }
            }

            return {};
        }
};

/*
This is the main function which doesn't do anything,
the functions/classes above will be answers to NeetCode questions
*/
int main()
{
    cout<<"Neet Code Grind";
    return 0;
}