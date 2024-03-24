import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

class ListNode {
  int val;
  ListNode next;

  ListNode() {
  }

  ListNode(int val) {
    this.val = val;
  }

  ListNode(int val, ListNode next) {
    this.val = val;
    this.next = next;
  }
}

class TreeNode {

  int val;

  TreeNode left;

  TreeNode right;

  TreeNode() {
  }

  TreeNode(int val) {
    this.val = val;
  }

  TreeNode(int val, TreeNode left, TreeNode right) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

class TrieNode {
  Map<Character, TrieNode> children;
  int count;

  public TrieNode() {
    this.children = new HashMap<>();
    this.count = 0;
  }
}

class Scratch {
  static int result = 0;
  int n;

  Set<String> set = new HashSet<>();

  public static void main(String[] args) {
    System.out.println(minDistance("horse", "ros"));
  }

  public static ListNode detectCycle(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;

    while (fast != null && fast.next != null) {
      slow = slow.next;
      fast = fast.next.next;

      if (slow == fast) {
        slow = head;

        while (slow != fast) {
          slow = slow.next;
          fast = fast.next;
        }

        return slow;
      }
    }

    return null;
  }

  public static boolean hasCycle(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;

    while (fast != null && fast.next != null) {
      slow = slow.next;
      fast = fast.next.next;

      if (slow == fast) {
        return true;
      }
    }

    return false;
  }

  public static int minimumDeleteSum(String s1, String s2) {
    int n1 = s1.length();
    int n2 = s2.length();

    int[][] dp = new int[n1 + 1][n2 + 1];

    for (int i = 1; i <= n1; i++) {
      dp[i][0] = dp[i - 1][0] + s1.charAt(i - 1);
    }

    for (int j = 1; j <= n2; j++) {
      dp[0][j] = dp[0][j - 1] + s2.charAt(j - 1);
    }

    for (int i = 1; i <= n1; i++) {
      for (int j = 1; j <= n2; j++) {
        if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = Math.min(dp[i - 1][j] + s1.charAt(i - 1),
              dp[i][j - 1] + s2.charAt(j - 1));
        }
      }
    }

    return dp[n1][n2];
  }

  public static int minDistanceDeleteOperationFor2Strings(String word1, String word2) {
    // https://leetcode.com/problems/delete-operation-for-two-strings/
    int n1 = word1.length();
    int n2 = word2.length();

    int[][] dp = new int[n1 + 1][n2 + 1];

    for (int i = 1; i <= n1; i++) {
      for (int j = 1; j <= n2; j++) {
        if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    return ((n1 - dp[n1][n2]) + (n2 - dp[n1][n2]));
  }

  public static int minDistance(String word1, String word2) {
    int n1 = word1.length();
    int n2 = word2.length();

    int[][] dp = new int[n1 + 1][n2 + 1];

    for (int i = 1; i <= n1; i++) {
      dp[i][0] = i;
    }

    for (int j = 1; j <= n2; j++) {
      dp[0][j] = j;
    }

    for (int i = 1; i <= n1; i++) {
      for (int j = 1; j <= n2; j++) {
        if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = Math.min(dp[i - 1][j - 1],
              Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
        }
      }
    }

    return dp[n1][n2];
  }

  public static int maxUncrossedLines(int[] nums1, int[] nums2) {
    // the basic algorithm for Longest Common Subsequence
    int n1 = nums1.length;
    int n2 = nums2.length;

    int[][] dp = new int[n1 + 1][n2 + 1];

    for (int i = 1; i <= n1; i++) {
      for (int j = 1; j <= n2; j++) {
        if (nums1[i - 1] == nums2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
        }
      }
    }

    return dp[n1][n2];
  }

  public static void reorderList(ListNode head) {
    // step 1. find the middle
    ListNode slow = head;
    ListNode fast = head;
    // to avoid cycle
    ListNode prev = head;

    while (fast != null && fast.next != null) {
      prev = slow;
      slow = slow.next;
      fast = fast.next.next;
    }
    prev.next = null;

    // step 2. reverse the second half
    ListNode first = head;
    ListNode second = reverseLL(slow);

    // step 3. merge
    mergeLL(first, second);

  }

  public static void mergeLL(ListNode first, ListNode second) {
    while (second != null) {
      ListNode next = first.next;
      first.next = second;
      first = second;
      second = next;
    }
  }

  public static ListNode reverseLL(ListNode head) {
    if (head == null)
      return null;
    ListNode prev = null;
    ListNode current = head;
    ListNode next = null;

    while (current != null) {
      next = current.next;
      current.next = prev;
      prev = current;
      current = next;
    }

    head = prev;
    return prev;
  }

  public static int rearrangeCharacters(String s, String target) {
    int[] freqS = new int[26];
    for (char c : s.toCharArray()) {
      freqS[c - 'a']++;
    }

    int[] freqTarget = new int[26];
    for (char c : target.toCharArray()) {
      freqTarget[c - 'a']++;
    }

    int max = Integer.MAX_VALUE;
    for (char c : target.toCharArray()) {
      if (freqTarget[c - 'a'] == 0) {
        return 0;
      }
      max = Math.min(max, freqS[c - 'a'] / freqTarget[c - 'a']);
    }

    return max;
  }

  /**
   * Inserts a new interval into a list of existing intervals.
   *
   * @param intervals   Array of intervals. Each interval is represented
   *                    by an array with two elements: the start and end
   *                    of the interval.
   * @param newInterval The new interval to be inserted.
   * @return Array of intervals with the new interval inserted.
   */
  public static int[][] insert(int[][] intervals, int[] newInterval) {
    // Initialize a counter to keep track of the current interval
    int i = 0;

    // Initialize a list to hold the result
    List<int[]> result = new ArrayList<>();

    // Iterate through the existing intervals
    while (i < intervals.length) {
      // Check if the current interval ends before the new interval starts
      if (intervals[i][1] < newInterval[0]) {
        // If so, add the current interval to the result
        result.add(intervals[i]);
      } else if (intervals[i][0] > newInterval[1]) {
        // If the current interval starts after the new interval ends,
        // break the loop since the new interval should be inserted after this one
        break;
      } else {
        // If the current interval and new interval overlap, update the
        // start and end of the new interval
        newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
        newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
      }
      // Increment the counter
      i++;
    }

    // Add the new interval to the result
    result.add(newInterval);

    // Add the remaining intervals to the result
    while (i < intervals.length) {
      result.add(intervals[i++]);
    }

    // Convert the list of intervals to a 2D array and return the result
    return result.toArray(new int[result.size()][2]);
  }

  public ListNode oddEvenList(ListNode head) {
    if (head == null || head.next == null
        || head.next.next == null) {
      return head;
    }

    ListNode oddHead = head;
    ListNode evenHead = head.next;
    ListNode oddTail = oddHead;
    ListNode evenTail = evenHead;

    ListNode curr = evenHead.next;
    boolean isOdd = true;

    while (curr != null) {
      ListNode next = curr.next;
      if (isOdd) {
        oddTail.next = curr;
        oddTail = curr;
      } else {
        evenTail.next = curr;
        evenTail = curr;
      }

      curr = next;
      isOdd = !isOdd;
    }

    oddTail.next = evenHead;
    evenTail.next = null;

    return oddHead;
  }

  public static int findMaxLength(int[] nums) {
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, -1);
    int zero = 0;
    int one = 0;
    int max = 0;

    for (int i = 0; i < nums.length; i++) {
      if (nums[i] == 0) {
        zero++;
      } else {
        one++;
      }

      int diff = zero - one;

      if (map.containsKey(diff)) {
        max = Math.max(max, i - map.get(diff));
      } else {
        map.put(diff, i);
      }
    }

    return max;
  }

  public static String getHappyString(int n, int k) {
    char[] pool = { 'a', 'b', 'c' };
    // dfsHappyString(n, k, 0, new char[n], pool);
    return "";

  }

  public String customSortString(String order, String s) {
    Character[] result = new Character[s.length()];
    for (int i = 0; i < s.length(); i++) {
      result[i] = s.charAt(i);
    }

    Arrays.sort(result, (c1, c2) -> {
      return order.indexOf(c1) - order.indexOf(c2);
    });

    String ans = "";
    for (Character c : result) {
      ans += c;
    }

    return ans;

  }

  public static boolean searchMatrix(int[][] matrix, int target) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
      return false;
    }

    int rows = matrix.length;
    int cols = matrix[0].length;

    int left = 0;
    int right = rows * cols - 1;

    while (left <= right) {
      int pivot = left + (right - left) / 2;
      int elem = matrix[pivot / cols][pivot % cols];
      if (elem == target) {
        return true;
      } else if (elem < target) {
        left = pivot + 1;
      } else {
        right = pivot - 1;
      }
    }

    return false;
  }

  public static List<Integer> intersection(int[][] nums) {
    List<Integer> result = new ArrayList<>();

    int[] freq = new int[1001];
    for (int[] arr : nums) {
      for (int i : arr) {
        freq[i]++;
      }
    }

    for (int i = 0; i < freq.length; i++) {
      if (freq[i] == nums.length) {
        result.add(i);
      }
    }

    return result;
  }

  public static List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
    Set<Integer> set1 = new HashSet<>();
    Set<Integer> set2 = new HashSet<>();
    List<List<Integer>> result = new ArrayList<>();

    for (int num : nums1) {
      set1.add(num);
    }

    for (int num : nums2) {
      set2.add(num);
    }

    List<Integer> distinct1 = new ArrayList<>(set1);
    distinct1.removeAll(set2);

    List<Integer> distinct2 = new ArrayList<>(set2);
    distinct2.removeAll(set1);

    result.add(distinct1);
    result.add(distinct2);

    return result;
  }

  public static int[] intersection(int[] nums1, int[] nums2) {
    Set<Integer> set = new HashSet<>();
    Set<Integer> intersection = new HashSet<>();
    for (int num : nums1) {
      set.add(num);
    }
    for (int num : nums2) {
      if (set.contains(num)) {
        intersection.add(num);
      }
    }
    return intersection.stream().mapToInt(Integer::valueOf).toArray();
  }

  public static int robb(int[] nums) {
    if (nums.length == 0)
      return 0;

    int dp1 = 0;
    int dp2 = 0;

    for (int num : nums) {
      int temp = dp1;
      dp1 = Math.max(dp2 + num, dp1);
      dp2 = temp;
    }

    return dp1;
  }

  public int numTrees(int n) {
    // horribleeeeeeeeee
    int[] dp = new int[n + 1];
    Arrays.fill(dp, 1);
    for (int i = 2; i <= n; i++) {
      for (int j = 1; j < i; j++) {
        dp[i] += dp[j] * dp[i - j];
      }
    }
    return dp[n];
  }

  public String findDifferentBinaryString(String[] nums) {
    n = nums.length;
    set.addAll(Arrays.asList(nums));
    return generateTheString("");
  }

  public String generateTheString(String curr) {
    if (curr.length() == n) {
      if (!set.contains(curr)) {
        return curr;
      }

      return "";
    }

    String zero = generateTheString(curr + "0");
    if (zero.length() > 0)
      return zero;

    return generateTheString(curr + "1");
  }

  public static int getWinner(int[] arr, int k) {
    int winner = arr[0];
    int wins = 0;
    for (int i = 1; i < arr.length; i++) {
      if (arr[i] > winner) {
        winner = arr[i];
        wins = 1;
      } else {
        wins++;
      }

      if (wins == k) {
        return winner;
      }
    }

    return winner;
  }

  public static int countNicePairs(int[] nums) {
    int count = 0;
    Map<Integer, Integer> map = new HashMap<>();
    for (int num : nums) {
      int rev = rev(num);
      int diff = map.getOrDefault(num - rev, 0);
      count = (count + diff) % 1000000007;
      map.put(num - rev, diff + 1);
    }

    return count;
  }

  public static int rev(int number) {
    int revNum = 0;
    while (number > 0) {
      revNum = revNum * 10 + number % 10;
      number /= 10;
    }
    return revNum;
  }

  public int countSegments(String s) {
    String trimmed = s.trim();
    if (trimmed.isEmpty())
      return 0;

    return trimmed.split("\\s+").length;
  }

  // public int longestPalindrome(String s) {
  // int[] arr = new int[128];
  // for (int i = 0; i < s.length(); i++) {
  // arr[s.charAt(i)]++;
  // }
  //
  // int ans = 0;
  // for (int i = 0; i < arr.length; i++) {
  // if (arr[i] % 2 == 0) {
  // ans += arr[i];
  // } else {
  // ans += arr[i] - 1;
  // }
  // }
  //
  // return ans < s.length() ? ans + 1 : ans;
  //
  // }
  //
  // public int countSubstrings(String s) {
  // int ans = 0;
  //
  // for (int i = 0; i < s.length(); i++) {
  // for (int j = i; j < s.length(); j++) {
  // if (isPalindrome(s, i, j)) ans++;
  // }
  // }
  //
  // return ans;
  // }

  public static int longestCommonSubsequence(String text1, String text2) {
    int l1 = text1.length();
    int l2 = text2.length();

    int[][] dp = new int[l1 + 1][l2 + 2];

    for (int i = 1; i <= l1; i++) {
      for (int j = 1; j <= l2; j++) {
        if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    return dp[l1][l2];
  }

  public static int maxLength(List<String> arr) {
    if (arr == null || arr.size() == 0)
      return 0;

    dfsMaxLength(arr, "", 0);

    return result;
  }

  public static void dfsMaxLength(List<String> arr, String path, int index) {
    boolean unique = isUnique(path);

    if (unique) {
      result = Math.max(path.length(), result);
    }

    if (index == arr.size() || !unique)
      return;

    for (int i = index; i < arr.size(); i++) {
      dfsMaxLength(arr, path + arr.get(i), i + 1);
    }
  }

  public static boolean isUnique(String s) {
    Set<Character> set = new HashSet<>();
    for (char c : s.toCharArray()) {
      if (set.contains(c))
        return false;
      set.add(c);
    }

    return true;
  }

  public static int[] findErrorNums(int[] nums) {
    int[] arr = new int[2];

    for (int i = 0; i < nums.length; i++) {
      int index = Math.abs(nums[i]) - 1;

      if (nums[index] < 0) {
        arr[0] = Math.abs(nums[i]);
      } else {
        nums[index] = -nums[index];
      }
    }

    for (int i = 0; i < nums.length; i++) {
      if (nums[i] > 0) {
        arr[1] = i + 1;
      }
    }

    return arr;
  }

  public int minimumPushes(String word) {
    int[] freq = new int[26];

    for (char c : word.toCharArray()) {
      freq[c - 'a']++;
    }

    Arrays.sort(freq);
    reverseArray(freq);

    int ans = 0;

    for (int i = 0; i < 26; i++) {
      int cost = i / 8 + 1;
      ans += cost * freq[i];
    }

    return ans;
  }

  public void reverseArray(int[] arr) {
    int start = 0;
    int end = arr.length - 1;

    while (start < end) {
      int temp = arr[start];
      arr[start] = arr[end];
      arr[end] = temp;
      start++;
      end--;
    }
  }

  public int rob(int[] nums) {
    // TODO houserobber1
    // https://leetcode.com/problems/house-robber/description/?envType=daily-question&envId=2024-01-21
    return 0;
  }

  public int sumSubarrayMins(int[] arr) {
    // TODO
    // https://leetcode.com/problems/sum-of-subarray-minimums/description/?envType=daily-question&envId=2024-01-20
    return 0;
  }

  public static boolean uniqueOccurrences(int[] arr) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int num : arr) {
      map.put(num, map.getOrDefault(num, 0) + 1);
    }

    Set<Integer> set = new HashSet<>(map.values());

    return map.size() == set.size();
  }

  public List<Integer> largestValues(TreeNode root) {
    List<Integer> ans = new ArrayList<>();

    if (root == null)
      return ans;

    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);

    while (!queue.isEmpty()) {
      int rowSize = queue.size();
      int max = Integer.MIN_VALUE;

      for (int i = 0; i < rowSize; i++) {
        TreeNode current = queue.poll();
        max = Math.max(max, current.val);

        if (current.left != null)
          queue.offer(current.left);
        if (current.right != null)
          queue.offer(current.right);
      }

      ans.add(max);
    }

    return ans;
  }

  public boolean findTarget(TreeNode root, int k) {
    List<Integer> nodes = new ArrayList<>();
    helperInorder(root, nodes);

    int left = 0;
    int right = nodes.size() - 1;

    while (left < right) {
      int current = nodes.get(left) + nodes.get(right);
      if (current == k) {
        return true;
      } else if (current < k) {
        left++;
      } else {
        right--;
      }
    }

    return false;
  }

  public static List<List<Integer>> findWinners(int[][] matches) {
    int[] losses = new int[100001];

    for (int i = 0; i < matches.length; i++) {
      int win = matches[i][0];
      int loss = matches[i][1];

      if (losses[win] == 0)
        losses[win] = -1;
      if (losses[loss] == -1) {
        losses[loss] = 1;
      } else {
        losses[loss]++;
      }
    }

    List<Integer> zeroLoss = new ArrayList<>();
    List<Integer> oneLoss = new ArrayList<>();

    List<List<Integer>> ans = new ArrayList<>();

    for (int i = 0; i < losses.length; i++) {
      if (losses[i] == -1) {
        zeroLoss.add(i);
      } else if (losses[i] == 1) {
        oneLoss.add(i);
      }
    }

    ans.add(zeroLoss);
    ans.add(oneLoss);

    return ans;
  }

  public List<Integer> beautifulIndices(String s, String a, String b, int k) {
    // USE KNUTH-MORRIS-PRATT FOR FINDING PATTERNS IN STRINGS
    int[] pa = prep((a + '#' + s).toCharArray());
    int[] pb = prep((b + '#' + s).toCharArray());
    List<Integer> ia = new ArrayList<>();
    List<Integer> ib = new ArrayList<>();

    for (int i = 0; i < pa.length; i++) {
      if (pa[i] == a.length()) {
        ia.add(i - a.length() * 2);
      }
    }

    for (int i = 0; i < pb.length; i++) {
      if (pb[i] == b.length()) {
        ib.add(i - b.length() * 2);
      }
    }

    List<Integer> ans = new ArrayList<>();
    for (int i : ia) {
      int p = binarySearch(ib, i);
      for (int p1 = p - 1; p1 <= p + 1; p1++) {
        if (0 <= p1 && p1 < ib.size() && Math.abs(ib.get(p1) - i) <= k) {
          ans.add(i);
          break;
        }
      }
    }

    return ans;
  }

  private int binarySearch(List<Integer> ib, int target) {
    int left = 0;
    int right = ib.size() - 1;

    while (left <= right) {
      int mid = left + (right - left) / 2;
      if (ib.get(mid) < target) {
        left = mid + 1;
      } else if (ib.get(mid) > target) {
        right = mid - 1;
      } else {
        return mid;
      }
    }

    return left;
  }

  private int[] prep(char[] p) {
    int[] pi = new int[p.length];
    int j = 0;

    for (int i = 1; i < p.length; i++) {
      while (j != 0 && p[j] != p[i]) {
        j = pi[j - 1];
      }

      if (p[j] == p[i]) {
        j++;
      }

      pi[i] = j;
    }

    return pi;
  }

  public static int maxFrequencyElements(int[] nums) {
    int[] freq = new int[10000];
    for (int num : nums) {
      freq[num]++;
    }

    int max = 0;
    for (int i : freq)
      max = Math.max(i, max);

    int ans = 0;
    for (int i = 1; i < freq.length; i++) {
      if (freq[i] == max) {
        ans += freq[i];
      }
    }

    return ans;
  }

  public static int maxOperations(int[] nums, int k) {
    Arrays.sort(nums);
    int start = 0;
    int end = nums.length - 1;
    int ans = 0;

    while (start < end) {
      if (nums[start] + nums[end] == k) {
        ans++;
        start++;
        end--;
      } else if (nums[start] + nums[end] > k) {
        end--;
      } else
        start++;
    }

    return ans;
  }

  public static String reverseWords(String s) {
    String[] words = s.split("\\s+");
    StringBuilder reversedString = new StringBuilder();

    for (int i = words.length - 1; i >= 0; i--) {
      if (!words[i].isEmpty()) {
        reversedString.append(words[i]).append(" ");
      }
    }

    return reversedString.toString().trim();
  }

  public static int distinctPrimeFactors(int[] nums) {
    Set<Integer> distinctPrimeFactors = new HashSet<>();

    for (int num : nums) {
      findDistinctPrimeFactors(num, distinctPrimeFactors);
    }

    return distinctPrimeFactors.size();
  }

  private static void findDistinctPrimeFactors(int num, Set<Integer> distinctPrimeFactors) {
    // Handle the case of negative numbers or 0
    if (num <= 0) {
      return;
    }

    // Find and add the distinct prime factors
    for (int i = 2; i * i <= num; i++) {
      while (num % i == 0) {
        distinctPrimeFactors.add(i);
        num /= i;
      }
    }

    // If num is a prime number greater than 1
    if (num > 1) {
      distinctPrimeFactors.add(num);
    }
  }

  public static int[] closestDivisors(int num) {
    for (int i = (int) Math.sqrt(num + 2); i > 0; --i) {
      if ((num + 1) % i == 0)
        return new int[] { i, (num + 1) / i };
      if ((num + 2) % i == 0)
        return new int[] { i, (num + 2) / i };
    }
    return new int[] {};
  }

  public static int findLucky(int[] arr) {
    int[] freq = new int[10000];

    for (int j : arr) {
      freq[j]++;
    }

    for (int i = arr.length; i > 0; i--) {
      if (freq[i] == i)
        return i;
    }

    return -1;
  }

  public static boolean kLengthApart(int[] nums, int k) {
    int count = 0;
    if (nums[0] == 0)
      count = k;

    for (int i = 1; i < nums.length; i++) {
      if (nums[i] == 1) {
        if (count < k)
          return false;
        count = 0;
      } else {
        count++;
      }
    }

    return true;
  }

  public static int minOperations(int[] nums) {
    int count = 0;
    int m2max = 0;

    for (int n : nums) {
      int m2 = 0;

      while (n > 1) {
        if (n % 2 == 1)
          count++;
        m2++;
        n /= 2;
      }

      if (n == 1)
        count++;
      m2max = Math.max(m2max, m2);
    }

    return m2max + count;
  }

  public static ListNode sortList(ListNode head) {
    if (head == null || head.next == null)
      return head;

    ListNode middle = middleNode(head);
    ListNode secondHead = middle.next;
    middle.next = null;

    ListNode l1 = sortList(head);
    ListNode l2 = sortList(secondHead);

    return mergeLinkedList(l1, l2);
  }

  public static ListNode mergeLinkedList(ListNode l1, ListNode l2) {
    ListNode head;

    if (l1 == null)
      return l2;
    if (l2 == null)
      return l1;

    if (l1.val < l2.val) {
      head = l1;
      l1 = l1.next;
    } else {
      head = l2;
      l2 = l2.next;
    }

    ListNode dummy = head;

    while (l1 != null && l2 != null) {
      if (l1.val < l2.val) {
        dummy.next = l1;
        l1 = l1.next;
        dummy = dummy.next;
      } else {
        dummy.next = l2;
        l2 = l2.next;
        dummy = dummy.next;
      }
    }

    if (l1 == null)
      dummy.next = l2;
    if (l2 == null)
      dummy.next = l1;

    return head;
  }

  public static String multiply(String num1, String num2) {
    // uglee Karatsuba
    if (num1 == null || num2 == null)
      return null;
    if ("".equals(num1) || "".equals(num2))
      return "";
    if (num1.length() == 1 || num2.length() == 1)
      return singleMultiply(num2, num1);

    int len = Math.max(num1.length(), num2.length());
    StringBuilder sb1 = new StringBuilder(num1);
    StringBuilder sb2 = new StringBuilder(num2);
    for (int a = sb1.length(); a < len; a++)
      sb1.insert(0, '0');
    for (int a = sb2.length(); a < len; a++)
      sb2.insert(0, '0');
    num1 = sb1.toString();
    num2 = sb2.toString();

    String[] fComp = split(num1), sComp = split(num2);
    String a = fComp[0], b = fComp[1];
    String c = sComp[0], d = sComp[1];

    String a_c = multiply(a, c);
    String b_d = multiply(b, d);
    String ab_cd = multiply(add(a, b), add(c, d));
    String e = subtract(subtract(ab_cd, b_d), a_c);
    return add(add(pad(a_c, (len >> 1) << 1), pad(e, len >> 1)), b_d);
  }

  private static String trim(StringBuilder sb) {
    while (sb.length() > 1 && sb.charAt(0) == '0')
      sb.deleteCharAt(0);
    return sb.toString();
  }

  private static String[] split(String s) {
    int m = s.length() >> 1;
    m = (s.length() & 1) == 1 ? m + 1 : m;
    return new String[] { s.substring(0, m), s.substring(m) };
  }

  private static String singleMultiply(String num1, String num2) {
    if ("0".equals(num1) || "0".equals(num2))
      return "0";
    if ("1".equals(num1))
      return num2;
    if ("1".equals(num2))
      return num1;

    String a, b;
    if (num1.length() == 1) {
      a = num2;
      b = num1;
    } else {
      a = num1;
      b = num2;
    }

    StringBuilder sb = new StringBuilder();
    int c = 0, s = b.charAt(0) - '0';

    for (int i = a.length() - 1; i >= 0; i--) {
      int f = a.charAt(i) - '0';
      sb.insert(0, ((f * s) + c) % 10);
      c = ((f * s) + c) / 10;
    }

    if (c > 0)
      sb.insert(0, c);
    return trim(sb);
  }

  private static String add(String num1, String num2) {
    String a, b;
    if (num1.length() >= num2.length()) {
      a = num1;
      b = num2;
    } else {
      a = num2;
      b = num1;
    }

    StringBuilder sb = new StringBuilder();
    int c = 0, diff = a.length() - b.length();

    for (int i = a.length() - 1; i >= 0; i--) {
      int f = a.charAt(i) - '0';
      int s = i - diff < 0 ? 0 : b.charAt(i - diff) - '0';
      sb.insert(0, (f + c + s) % 10);
      c = (f + c + s) / 10;
    }

    if (c > 0)
      sb.insert(0, c);
    return trim(sb);
  }

  private static String subtract(String num1, String num2) {
    String a = null, b = null;

    if (num1.length() > num2.length()) {
      a = num1;
      b = num2;
    } else if (num1.length() < num2.length()) {
      a = num2;
      b = num1;
    } else {

      for (int i = 0; i < num1.length(); i++) {
        if (num1.charAt(i) - '0' > num2.charAt(i) - '0') {
          a = num1;
          b = num2;
          break;
        } else if (num2.charAt(i) - '0' > num1.charAt(i) - '0') {
          a = num2;
          b = num1;
          break;
        }
      }
      if (a == null)
        return "0";
    }

    StringBuilder sb = new StringBuilder();
    int c = 0, diff = a.length() - b.length();

    for (int i = a.length() - 1; i >= 0; i--) {
      int f = a.charAt(i) - '0';
      int s = i - diff < 0 ? 0 : b.charAt(i - diff) - '0';
      int p = f - c - s;
      if (p < 0) {
        p += 10;
        c = 1;
      } else {
        c = 0;
      }
      sb.insert(0, p);
    }

    return trim(sb);
  }

  private static String pad(String num, int zeros) {
    StringBuilder sb = new StringBuilder(num);
    for (int a = 0; a < zeros; a++)
      sb.append('0');
    return sb.toString();
  }

  public static int longestSubsequence(String s, int k) {
    char[] arr = s.toCharArray();
    int zeroes = 0;

    for (char c : arr) {
      if (c == '0')
        zeroes++;
    }

    int ones = 0;
    int val = 0;
    int n = arr.length;
    for (int i = n - 1; i >= 0; i--) {
      int x = (int) (val + Math.pow(2, n - 1 - i) * (arr[i] - '0'));
      if (x <= k) {
        val += Math.pow(2, n - 1 - i) * (arr[i] - '0');
        if (arr[i] == '1') {
          ones++;
        }
      } else {
        break;
      }
    }

    return zeroes + ones;
  }

  // TLE too
  // https://leetcode.com/problems/longest-binary-subsequence-less-than-or-equal-to-k/description/
  // public static int longestSubsequence(String s, int k) {
  // if(s == null || k == 0) return 0;
  //
  // Map<String, Integer> memo = new HashMap<>();
  // String longest = s.substring(0, 1);
  // for(int i = 0; i < s.length(); i++){
  // String temp = longestSubsequenceHelper(s, i, k, "", memo);
  // if(temp.length() > longest.length()) longest = temp;
  // }
  //
  // return longest.length();
  // }
  //
  // private static String longestSubsequenceHelper(String s, int i, int k, String
  // current,
  // Map<String, Integer> memo) {
  // if(i == s.length()){
  // if(current.isEmpty() || Integer.parseInt(current, 2) <= k){
  // return current;
  // } else {
  // return "";
  // }
  // }
  //
  // String key = i + ":" + current;
  // if(memo.containsKey(key)) {return memo.get(key).toString();}
  //
  // String includeCurrent = longestSubsequenceHelper(s, i + 1, k, current +
  // s.charAt(i),
  // memo);
  // String excludeCurrent = longestSubsequenceHelper(s, i + 1, k, current, memo);
  //
  // String ans = (includeCurrent.length() > excludeCurrent.length()) ?
  // includeCurrent :
  // excludeCurrent;
  // memo.put(key, ans.length());
  // return ans;
  // }

  // TLE so we do memoization
  // public static int longestSubsequence(String s, int k) {
  // if(s == null || k == 0) return 0;
  //
  // String longest = s.substring(0, 1);
  // for(int i = 0; i < s.length(); i++){
  // String temp = longestSubsequenceHelper(s, i, k, "");
  // if(temp.length() > longest.length()) longest = temp;
  // }
  //
  // return longest.length();
  // }
  //
  // private static String longestSubsequenceHelper(String s, int i, int k, String
  // current) {
  // if(i == s.length()){
  // if(current.isEmpty() || Integer.parseInt(current, 2) <= k){
  // return current;
  // } else {
  // return "";
  // }
  // }
  //
  // String includeCurrent = longestSubsequenceHelper(s, i + 1, k, current +
  // s.charAt(i));
  // String excludeCurrent = longestSubsequenceHelper(s, i + 1, k, current);
  //
  // return (includeCurrent.length() > excludeCurrent.length()) ? includeCurrent :
  // excludeCurrent;
  // }

  public boolean closeStrings(String word1, String word2) {
    int[] first = new int[26];
    int[] second = new int[26];

    for (char c : word1.toCharArray())
      first[c - 'a']++;
    for (char c : word2.toCharArray())
      second[c - 'a']++;

    int ans = 0;

    for (int i = 0; i < 26; i++) {
      if ((first[i] == 0 && second[i] != 0) || (first[i] != 0 && second[i] == 0)) {
        return false;
      }
    }

    Arrays.sort(first);
    Arrays.sort(second);

    for (int i = 0; i < 26; i++) {
      if (first[i] != second[i])
        return false;
    }

    return true;
  }

  public static int minStepsII(String s, String t) {
    int[] first = new int[26];
    int[] second = new int[26];

    for (char c : s.toCharArray())
      first[c - 'a']++;
    for (char c : t.toCharArray())
      second[c - 'a']++;

    int ans = 0;

    for (int i = 0; i < 26; i++) {
      ans += Math.abs(first[i] - second[i]);
    }

    return ans;
  }

  public int minStepsI(String s, String t) {
    int[] first = new int[26];
    int[] second = new int[26];

    for (char c : s.toCharArray())
      first[c - 'a']++;
    for (char c : t.toCharArray())
      second[c - 'a']++;

    int ans = 0;

    for (int i = 0; i < 26; i++) {
      ans += Math.abs(first[i] - second[i]);
    }

    return ans / 2;
  }

  public static boolean halvesAreAlike(String s) {
    int first = 0;
    int second = 0;
    Set<Character> vowels = new HashSet<>();
    vowels.add('a');
    vowels.add('e');
    vowels.add('i');
    vowels.add('o');
    vowels.add('u');
    vowels.add('A');
    vowels.add('E');
    vowels.add('I');
    vowels.add('O');
    vowels.add('U');

    for (int i = 0; i < s.length() / 2; i++) {
      if (vowels.contains(s.charAt(i)))
        first++;
    }

    for (int i = s.length() / 2; i < s.length(); i++) {
      if (vowels.contains(s.charAt(i)))
        second++;
    }

    return first == second;
  }

  public int diff = 0;

  public int maxAncestorDiff(TreeNode root) {
    if (root == null)
      return 0;

    int min = root.val;
    int max = root.val;

    diff(root, min, max);

    return diff;
  }

  public void diff(TreeNode root, int min, int max) {
    if (root == null)
      return;

    diff = Math.max(diff, Math.max(Math.abs(min - root.val), Math.abs(max - root.val)));
    min = Math.min(min, root.val);
    max = Math.max(max, root.val);

    diff(root.left, min, max);
    diff(root.right, min, max);
  }

  public static boolean isLongPressedName(String name, String typed) {
    int i = 0, j = 0;

    while (j < typed.length()) {
      if (i < name.length() && name.charAt(i) == typed.charAt(j)) {
        i++;
        j++;
      } else if (j > 0 && typed.charAt(j) == typed.charAt(j - 1)) {
        j++;
      } else {
        return false;
      }
    }

    return i == name.length();
  }

  public static int matchPlayersAndTrainers(int[] players, int[] trainers) {
    if (trainers.length == 1 && players.length == 1 && players[0] <= trainers[0])
      return 1;

    Arrays.sort(players);
    Arrays.sort(trainers);

    int count = 0;

    int i = 0;
    int j = 0;

    while (i < players.length && j < trainers.length) {
      if (players[i] <= trainers[j]) {
        count++;
        i++;
        j++;
      } else if (players[i] > trainers[j]) {
        j++;
      }
    }

    return count;
  }

  public int rangeSumBST(TreeNode root, int low, int high) {
    if (root == null)
      return 0;

    int current = (root.val >= low && root.val <= high) ? root.val : 0;
    int left = rangeSumBST(root.left, low, high);
    int right = rangeSumBST(root.right, low, high);

    return current + left + right;
  }

  public int numberOfArithmeticSlices(int[] nums) {
    // https://leetcode.com/problems/arithmetic-slices-ii-subsequence/description/
    // THE FIRST ONE TOO
    // https://leetcode.com/problems/arithmetic-slices/description/
    int n = nums.length;
    int total_count = 0;

    Map<Integer, Integer>[] dp = new HashMap[n];

    for (int i = 0; i < n; ++i) {
      dp[i] = new HashMap<>();
    }

    for (int i = 1; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        long diff = (long) nums[i] - nums[j];

        if (diff > Integer.MAX_VALUE || diff < Integer.MIN_VALUE) {
          continue;
        }

        int diffInt = (int) diff;

        dp[i].put(diffInt, dp[i].getOrDefault(diffInt, 0) + 1);
        if (dp[j].containsKey(diffInt)) {
          dp[i].put(diffInt, dp[i].get(diffInt) + dp[j].get(diffInt));
          total_count += dp[j].get(diffInt);
        }
      }
    }

    return total_count;
  }

  public static int maximumSetSize(int[] nums1, int[] nums2) {
    Set<Integer> set1 = new HashSet<>();
    Set<Integer> set2 = new HashSet<>();

    Set<Integer> overlap = new HashSet<>();

    int target = nums1.length / 2;

    for (int num : nums1) {
      set1.add(num);
    }

    for (int num : nums2) {
      set2.add(num);
    }

    for (int num : set1) {
      if (set2.contains(num))
        overlap.add(num);
    }

    for (int num : overlap) {
      if (set1.size() >= set2.size())
        set1.remove(num);
      else
        set2.remove(num);
    }

    return Math.min(set1.size(), target) + Math.min(set2.size(), target);
  }

  public int areaOfMaxDiagonal(int[][] dimensions) {
    // https://leetcode.com/contest/weekly-contest-379/problems/maximum-area-of-longest-diagonal-rectangle
    double maxDiagonal = 0;
    int maxArea = 0;

    for (int[] rectangle : dimensions) {
      int length = rectangle[0];
      int width = rectangle[1];

      double diagonal = Math.sqrt(length * length + width * width);

      if (diagonal > maxDiagonal || (diagonal == maxDiagonal && length * width > maxArea)) {
        maxDiagonal = diagonal;
        maxArea = length * width;
      }
    }

    return maxArea;
  }

  public boolean isNStraightHand(int[] hand, int groupSize) {
    // same as
    // https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/
    Map<Integer, Integer> map = new HashMap<>();
    for (int card : hand) {
      map.put(card, map.getOrDefault(card, 0) + 1);
    }

    Arrays.sort(hand);

    for (int k : hand) {
      if (map.get(k) == 0)
        continue;

      for (int j = 0; j < groupSize; j++) {
        int current = k + j;

        if (map.getOrDefault(current, 0) == 0)
          return false;

        map.put(current, map.get(current) - 1);
      }
    }

    return true;
  }

  public static int maxEvents(int[][] events) {
    // TODO IDK HAVEN'T SOLVED IT YET
    Arrays.sort(events, Comparator.comparingInt(a -> a[1]));

    PriorityQueue<Integer> heap = new PriorityQueue<>();
    int max = 0;
    int day = 1;
    int n = events.length;
    int event = 0;

    while (day <= 100000 && (event < n || !heap.isEmpty())) {
      // events for today
      while (event < n && events[event][0] == day) {
        heap.offer(events[event++][1]);
      }

      // remove what has already ended
      while (!heap.isEmpty() && heap.peek() < day) {
        heap.poll();
      }

      // attend
      if (!heap.isEmpty()) {
        heap.poll();
        max++;
      }

      day++;
    }

    return max;
  }

  public long maxTaxiEarnings(int n, int[][] rides) {
    Arrays.sort(rides, Comparator.comparingInt(a -> a[1]));
    long[] dp = new long[n + 1];
    int j = 0;

    for (int i = 1; i < dp.length; i++) {
      dp[i] = dp[i - 1];

      while (j < rides.length && i == rides[j][1]) {
        int[] ride = rides[j++];
        dp[i] = Math.max(dp[i], dp[ride[0]] + ride[1] - ride[0] + ride[2]);
      }
    }

    return dp[n];
  }

  public String largestNumber(int[] nums) {
    String[] arr = new String[nums.length];
    for (int i = 0; i < nums.length; i++) {
      arr[i] = String.valueOf(nums[i]);
    }

    StringBuilder sb = new StringBuilder();
    Arrays.sort(arr, (a, b) -> (b + a).compareTo(a + b));

    for (String s : arr) {
      sb.append(s);
    }

    String ans = sb.toString();

    return ans.startsWith("0") ? "0" : ans;
  }

  // public List<Boolean> prefixesDivBy5(int[] nums) {
  // //https://leetcode.com/problems/binary-prefix-divisible-by-5/description/
  // return new List<Boolean>{true, false};
  // }

  public static int averageValue(int[] nums) {
    int ans = 0;
    int count = 0;
    for (int i = 0; i < nums.length; i++) {
      if (nums[i] % 2 == 0 && nums[i] % 3 == 0) {
        ans += nums[i];
        count++;
      }
    }

    return count == 0 ? 0 : ans / count;
  }

  public static int maximumSum(int[] nums) {
    // 2342 TLE
    // int max = -1;
    //
    // for(int i = 0; i < nums.length; i++) {
    // for (int j = i + 1; j < nums.length; j++) {
    // if (sum(nums[i]) == sum(nums[j])) {
    // max = Math.max(max, nums[i] + nums[j]);
    // }
    // }
    // }
    // return max;

    Map<Integer, Integer> map = new HashMap<>();
    int max = -1;

    for (int num : nums) {
      int sum = sum(num);

      if (map.containsKey(sum)) {
        int other = map.get(sum);
        max = Math.max(max, num + other);
        map.put(sum, Math.max(other, num));
      } else {
        map.put(sum, num);
      }
    }

    return max;
  }

  public static int sum(int num) {
    int sum = 0;
    while (num > 0) {
      sum += num % 10;
      num /= 10;
    }

    return sum;
  }

  public static int minimumOperations(int[] nums) {
    Set<Integer> set = new HashSet<>();

    for (int num : nums) {
      if (num != 0)
        set.add(num);
    }

    return set.size();
  }

  public static int longestNiceSubarray(int[] nums) {
    int left = 0;
    int ans = 0;

    for (int right = 0; right < nums.length; right++) {
      while (left < right && check(nums, left, right))
        left++;

      ans = Math.max(ans, right - left + 1);
    }

    return ans;
  }

  public static boolean check(int[] nums, int left, int right) {
    for (int i = left; i < right; i++) {
      if ((nums[i] & nums[right]) != 0)
        return true;
    }

    return false;
  }

  public static long dividePlayers(int[] skill) {
    Arrays.sort(skill);
    long chemistry = 0;

    int firstPair = skill[0] + skill[skill.length - 1];

    for (int i = 0; i < skill.length / 2; i++) {
      int current = skill[i];
      int currentLast = skill[skill.length - 1 - i];

      // checking if total skill is the same
      if (firstPair != current + currentLast)
        return -1;
      chemistry += (long) currentLast * current;
    }

    return chemistry;
  }

  public static int pivotInteger(int n) {
    int left = 0;
    int right = n;
    int pivot;

    while (left <= right) {
      pivot = left + (right - left) / 2;

      long firstSum = (long) pivot * (pivot + 1) / 2;
      long secondSum = (long) (n - pivot + 1) * (n + pivot) / 2;

      if (firstSum == secondSum) {
        return pivot;
      } else {
        if (firstSum < secondSum) {
          left = pivot + 1;
        } else {
          right = pivot - 1;
        }
      }
    }

    return -1;
  }

  public static boolean findSubarrays(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int i = 0; i < nums.length - 1; i++) {
      int sum = nums[i] + nums[i + 1];
      if (!set.add(sum))
        return true;
    }

    return false;
  }

  public int minimumRounds(int[] tasks) {
    int ans = 0;
    Map<Integer, Integer> map = new HashMap<>();

    for (int num : tasks) {
      map.put(num, map.getOrDefault(num, 0) + 1);
    }

    for (int value : map.values()) {
      if (value == 1)
        return -1;

      ans += (int) Math.ceil((double) value / 3);
    }

    return ans;
  }

  public static int lengthOfLastWord(String s) {
    s = s.trim();
    if (s.isEmpty() || s.equals(" "))
      return 0;
    int length = 0;
    char[] arr = s.toCharArray();
    for (int i = arr.length - 1; i >= 0; i--) {
      if (arr[i] != ' ') {
        length++;
      } else {
        break;
      }
    }

    return length;
  }

  public static boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
    // optimal
    // if (nums == null || nums.length == 0 || k <= 0 || t < 0) {
    // return false;
    // }
    // TreeSet<Long> set = new TreeSet<>();
    // for (int i = 0; i < nums.length; i++) {
    // long num = (long) nums[i];
    // Long floor = set.floor(num);
    // if (floor != null && num - floor <= t) {
    // return true;
    // }
    // Long ceiling = set.ceiling(num);
    // if (ceiling != null && ceiling - num <= t) {
    // return true;
    // }
    // set.add(num);
    // if (i >= k) {
    // set.remove((long) nums[i - k]);
    // }
    // }
    // return false;
    // TLE
    int i = 0;
    boolean found = false;
    while (i < nums.length && !found) {
      int j = i + 1;
      while (j < nums.length && !found) {
        if (i != j && Math.abs(i - j) <= indexDiff && Math.abs(nums[i] - nums[j]) <= valueDiff) {
          found = true;
        }
        j++;
      }
      i++;
    }

    return found;
  }

  public int incremovableSubarrayCount(int[] nums) {
    return 0;
  }

  public static int maximumLength(String s) {
    int ans = -1;
    char[] arr = s.toCharArray();
    Map<String, Integer> map = new HashMap<>();

    for (int i = 0; i < arr.length; i++) {
      for (int j = i; j < arr.length; j++) {
        String current = s.substring(i, j + 1);
        map.put(current, map.getOrDefault(current, 0) + 1);
      }
    }

    for (String str : map.keySet()) {
      if (map.get(str) >= 3) {
        boolean flag = true;

        for (int i = 1; i < str.length() && flag; i++) {
          if (str.charAt(i) != str.charAt(0)) {
            flag = false;
          }
        }

        if (flag) {
          ans = Math.max(ans, str.length());
        }
      }
    }

    return ans;
  }

  public static boolean hasTrailingZeros(int[] nums) {
    int count = 0;
    for (int i = 0; i < nums.length; i++) {
      for (int j = 0; j < nums.length; j++) {
        int check = nums[i] | nums[j];
        String stringCheck = Integer.toBinaryString(check);
        System.out.println(
            "number " + nums[i] + " and number " + nums[j] + " =check is " + stringCheck);
        if (stringCheck.endsWith("00") || stringCheck.endsWith("0")) {
          count++;
          System.out.println("COUNTED FOR " + nums[i] + " AND " + nums[j]);
        }
      }
    }

    return count >= 2;
  }

  public static List<List<Integer>> findMatrix(int[] nums) {
    int[] freq = new int[nums.length + 1];
    List<List<Integer>> ans = new ArrayList<>();

    for (int num : nums) {
      if (freq[num] >= ans.size()) {
        ans.add(new ArrayList<>());
      }

      ans.get(freq[num]).add(num);
      freq[num]++;
    }

    return ans;
  }

  public static boolean wordBreak(String s, List<String> wordDict) {
    Set<String> set = new HashSet<>(wordDict);
    int n = s.length();
    boolean[] dp = new boolean[n + 1];
    dp[0] = true;

    for (int i = 1; i <= n; i++) {
      for (int j = 0; j < i; j++) {
        if (dp[j] && set.contains(s.substring(j, i))) {
          dp[i] = true;
          break;
        }
      }
    }

    return dp[n];
  }

  public static int minExtraChar(String s, String[] dictionary) {
    int[] dp = new int[s.length() + 1];
    Arrays.fill(dp, Integer.MAX_VALUE);
    dp[0] = 0;

    for (int i = 1; i <= s.length(); i++) {
      for (String word : dictionary) {
        if (i >= word.length() && s.startsWith(word, i - word.length())) {
          dp[i] = Math.min(dp[i], dp[i - word.length()]);
        }
      }

      dp[i] = Math.min(dp[i], dp[i - 1] + 1);
    }

    return dp[s.length()];
  }

  public static int bestClosingTime(String customers) {
    int y = 0;
    int n = 0;

    for (char c : customers.toCharArray()) {
      if (c == 'Y') {
        y++;
      } else {
        n++;
      }
    }

    int profit = n;
    int loss = y;
    int[] ans = new int[customers.length() + 1];
    ans[0] = profit - loss;

    for (int i = 1; i <= customers.length(); i++) {
      char c = customers.charAt(i - 1);
      if (c == 'Y') {
        profit++;
      } else {
        loss++;
      }

      ans[i] = profit - loss;
    }

    int iCopy = 0;
    int max = Integer.MIN_VALUE;
    for (int i = 0; i <= customers.length(); i++) {
      if (ans[i] > max) {
        max = ans[i];
        iCopy = i;
      }
    }

    return iCopy;
  }

  public static int findContentChildren(int[] greed, int[] cookies) {
    Arrays.sort(greed);
    Arrays.sort(cookies);

    int content = 0;
    int i = 0;
    int j = 0;

    while (i < greed.length && j < cookies.length) {
      if (cookies[j] >= greed[i]) {
        content++;
        i++;
      }
      j++;
    }

    return content;
  }

  public static boolean makeEqual(String[] words) {
    Map<Character, Integer> map = new HashMap<>();
    for (String word : words) {
      for (char c : word.toCharArray()) {
        map.put(c, map.getOrDefault(c, 0) + 1);
      }
    }

    for (char c : map.keySet()) {
      if (map.get(c) % words.length != 0) {
        return false;
      }
    }

    return true;
  }

  public static int[] numberGame(int[] nums) {
    Arrays.sort(nums);
    for (int i = 1; i < nums.length; i += 2) {
      int temp = nums[i];
      nums[i] = nums[i - 1];
      nums[i - 1] = temp;
    }

    return nums;
  }

  public int minDifficulty(int[] jobDifficulty, int d) {
    int n = jobDifficulty.length;
    if (d > n)
      return -1;
    int[][] dp = new int[d][n];
    for (int i = 1; i < d; i++) {
      Arrays.fill(dp[i], Integer.MAX_VALUE);
    }
    int maxDifficulty = 0;
    for (int i = 0; i <= n - d; i++) {
      maxDifficulty = Math.max(maxDifficulty, jobDifficulty[i]);
      dp[0][i] = maxDifficulty;
    }
    for (int i = 1; i < d; i++) {
      for (int j = i; j <= n - d + i; j++) {
        int currentDayDifficulty = jobDifficulty[j];
        int result = Integer.MAX_VALUE;
        for (int k = j - 1; k >= i - 1; k--) {
          result = Math.min(result, dp[i - 1][k] + currentDayDifficulty);
          currentDayDifficulty = Math.max(currentDayDifficulty, jobDifficulty[k]);
        }
        dp[i][j] = result;
      }
    }
    return dp[d - 1][n - 1];
  }

  public static List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();

    if (nums.length < 3)
      return ans;

    Arrays.sort(nums);

    for (int i = 0; i < nums.length - 2; i++) {
      // skip dps
      if (i > 0 && nums[i] == nums[i - 1])
        continue;

      int target = -nums[i];
      int left = i + 1;
      int right = nums.length - 1;

      while (left < right) {
        int sum = nums[left] + nums[right];

        if (sum == target) {
          ans.add(Arrays.asList(nums[i], nums[left], nums[right]));

          while (left < right && nums[left] == nums[left + 1])
            left++;
          while (left < right && nums[right] == nums[right - 1])
            right--;

          left++;
          right--;
        } else if (sum < target) {
          left++;
        } else {
          right--;
        }
      }
    }

    return ans;
  }

  public static boolean winnerSquareGame(int n) {
    int[] dp = new int[n + 1];
    for (int i = 1; i <= n; i++) {
      for (int j = 1; j * j <= i; j++) {
        dp[i] |= (dp[i - j * j] == 0) ? 1 : 0;
      }
    }

    return dp[n] == 1;
  }

  public static int numIslands(char[][] grid) {
    // bfs
    if (grid == null || grid.length == 0 || grid[0].length == 0) {
      return 0;
    }

    int rows = grid.length;
    int cols = grid[0].length;
    int numIslands = 0;

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (grid[i][j] == '1') {
          numIslands++;
          numberOfIslandsDfs(grid, i, j);
        }
      }
    }

    return numIslands;
  }

  public static void numberOfIslandsDfs(char[][] grid, int i, int j) {
    int rows = grid.length;
    int cols = grid[0].length;

    if (i < 0 || j < 0 || i >= rows || j >= cols || grid[i][j] == '0') {
      return;
    }

    grid[i][j] = '0'; // mark visited

    numberOfIslandsDfs(grid, i + 1, j);
    numberOfIslandsDfs(grid, i - 1, j);
    numberOfIslandsDfs(grid, i, j + 1);
    numberOfIslandsDfs(grid, i, j - 1);
  }

  public static boolean canFinish(int numCourses, int[][] prerequisites) {
    // https://leetcode.com/problems/course-schedule/
    return true;
  }

  // TLE 787. Cheapest Flights Within K Stops
  public static int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
    Map<Integer, List<int[]>> graph = new HashMap<>();

    // build the graph
    for (int[] flight : flights) {
      graph.putIfAbsent(flight[0], new ArrayList<>());
      graph.get(flight[0]).add(new int[] { flight[1], flight[2] });
    }

    // pq to store cost, current city, stops
    PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
    pq.offer(new int[] { 0, src, 0 });

    while (!pq.isEmpty()) {
      int[] current = pq.poll();
      int cost = current[0];
      int city = current[1];
      int stops = current[2];

      if (city == dst)
        return cost;

      if (stops <= k) {
        List<int[]> neighbors = graph.getOrDefault(city, new ArrayList<>());
        for (int[] neighbor : neighbors) {
          int nextCity = neighbor[0];
          int nextCost = cost + neighbor[1];
          int nextStops = stops + 1;
          pq.offer(new int[] { nextCost, nextCity, nextStops });
        }
      }
    }

    return -1;
  }

  public static long maximumImportance(int n, int[][] roads) {
    long ans = 0;
    long x = 1;
    long[] degree = new long[n];

    for (int[] road : roads) {
      degree[road[0]]++;
      degree[road[1]]++;
    }

    Arrays.sort(degree);

    for (long d : degree) {
      ans += d * (x++);
    }

    return ans;
  }

  // BFS
  public static List<List<Integer>> allPathsSourceTarget(int[][] graph) {
    List<List<Integer>> ans = new ArrayList<>();
    List<Integer> current = new ArrayList<>();
    allPathsSourceTargetDFS(0, graph, current, ans);

    return ans;
  }

  public static void allPathsSourceTargetDFS(
      int node, int[][] graph, List<Integer> current, List<List<Integer>> ans) {
    current.add(node);

    if (node == graph.length - 1) {
      ans.add(new ArrayList<>(current));
    } else {
      for (int neighbor : graph[node]) {
        allPathsSourceTargetDFS(neighbor, graph, current, ans);
      }
    }

    current.remove(current.size() - 1);
  }

  // GRAPHS - DIJKSTRA shortest path
  public static int countPaths(int n, int[][] roads) {
    List<int[]>[] graph = new ArrayList[n];
    for (int i = 0; i < n; i++) {
      graph[i] = new ArrayList<>();
    }

    for (int[] road : roads) {
      int u = road[0];
      int v = road[1];
      int time = road[2];

      graph[u].add(new int[] { v, time });
      graph[v].add(new int[] { u, time });
    }

    return dijkstra(n, graph);
  }

  public static int dijkstra(int n, List<int[]>[] graph) {
    long[] distance = new long[n];
    int[] ways = new int[n];
    Arrays.fill(distance, Long.MAX_VALUE);
    distance[0] = 0;
    ways[0] = 1;

    PriorityQueue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[1]));
    pq.offer(new long[] { 0, 0 });

    while (!pq.isEmpty()) {
      long[] current = pq.poll();
      int u = (int) current[0];
      long dist = current[1];

      if (dist > distance[u]) {
        continue;
      }

      for (int[] neighbor : graph[u]) {
        int v = neighbor[0];
        long time = neighbor[1];

        if (distance[u] + time < distance[v]) {
          distance[v] = distance[u] + time;
          ways[v] = ways[u];
          pq.offer(new long[] { v, distance[v] });

        } else if (distance[u] + time == distance[v]) {
          ways[v] = (ways[v] + ways[u]) % 1_000_000_007;
        }
      }
    }

    return ways[n - 1] % 1_000_000_007;
  }

  public static int garbageCollection(String[] garbage, int[] travel) {
    int ans = 0;
    Set<Character> set = new HashSet<>();
    for (int i = garbage.length - 1; i >= 0; i--) {
      for (char ch : garbage[i].toCharArray()) {
        set.add(ch);
      }

      ans += garbage[i].length();
      ans += i > 0 ? set.size() * travel[i - 1] : 0;
    }

    return ans;
  }

  public static int minCost(String colors, int[] neededTime) {
    int min = 0;
    for (int i = 1; i < colors.length(); i++) {
      if (colors.charAt(i) == colors.charAt(i - 1)) {
        min += Math.min(neededTime[i], neededTime[i - 1]);
        neededTime[i] = Math.max(neededTime[i], neededTime[i]);
      }
    }

    return min;
  }

  public static int numRollsToTarget(int n, int k, int target) {
    Map<String, Integer> memo = new HashMap<>();
    return numRollsToTargetHelper(n, k, target, memo);
  }

  public static int numRollsToTargetHelper(int n, int k, int target, Map<String, Integer> memo) {
    if (n == 0 && target == 0)
      return 1;
    if (n == 0 || target <= 0)
      return 0;

    String key = n + "," + target;
    if (memo.containsKey(key))
      return memo.get(key);

    int ans = 0;
    for (int i = 1; i <= k; i++) {
      ans = (ans + numRollsToTargetHelper(n - 1, k, target - i, memo)) % 1000000007;
    }

    memo.put(key, ans);
    return ans;
  }

  public List<List<Integer>> combinationSum3(int k, int n) {
    List<List<Integer>> ans = new ArrayList<>();
    helper(ans, new ArrayList<>(), k, n, 1);

    return ans;
  }

  public void helper(List<List<Integer>> ans, List<Integer> current, int k, int n, int start) {
    if (current.size() == k && n == 0) {
      ans.add(new ArrayList<>(current));
      return;
    }

    for (int i = start; i <= 9 && i <= n; i++) {
      current.add(i);
      helper(ans, current, k, n - i, i + 1);
      current.remove(current.size() - 1);
    }
  }

  public static boolean validMountainArray(int[] arr) {
    if (arr.length < 3)
      return false;

    int peak = 0;
    while (peak < arr.length - 1 && arr[peak] < arr[peak + 1]) {
      peak++;
    }

    if (peak == 0 || peak == arr.length - 1)
      return false;

    while (peak < arr.length - 1 && arr[peak] > arr[peak + 1]) {
      peak++;
    }

    return peak == arr.length - 1;
  }

  public static int[] replaceElements(int[] arr) {
    int max = -1;
    int end = arr.length - 1;

    while (end >= 0) {
      int temp = arr[end];
      arr[end] = max;
      if (temp > max) {
        max = temp;
      }

      end--;
    }

    return arr;
  }

  public static int minOperations(String s) {
    int start = 0;
    for (int i = 0; i < s.length(); i++) {
      if (i % 2 == 0) {
        if (s.charAt(i) == '1') {
          start++;
        }
      } else {
        if (s.charAt(i) == '0') {
          start++;
        }
      }
    }

    return Math.min(start, s.length() - start);
  }

  public static List<Integer> targetIndices(int[] nums, int target) {
    List<Integer> ans = new ArrayList<>();
    Arrays.sort(nums);

    for (int i = 0; i < nums.length; i++) {
      if (nums[i] == target)
        ans.add(i);
    }

    return ans;
  }

  public static List<Integer> findWordsContaining(String[] words, char x) {
    List<Integer> ans = new ArrayList<>();
    for (int i = 0; i < words.length; i++) {
      if (words[i].indexOf(x) != -1)
        ans.add(i);
    }

    return ans;
  }

  public static int maxScore(String s) {
    int ones = 0;
    int zeros = 0;
    int max = Integer.MIN_VALUE;

    for (int i = 0; i < s.length() - 1; i++) {
      if (s.charAt(i) == '1') {
        ones++;
      } else {
        zeros++;
      }

      max = Math.max(max, zeros - ones);
    }

    if (s.charAt(s.length() - 1) == '1') {
      ones++;
    }

    return max + ones;
  }

  public int pathSum3(TreeNode root, int targetSum) {
    // doesn't work for big numbers or something
    int[] ans = new int[1];
    hasPathSum3Helper(root, targetSum, new ArrayList<>(), ans);
    return ans[0];
  }

  public void hasPathSum3Helper(TreeNode root, int targetSum, List<Integer> current, int[] ans) {
    if (root == null)
      return;

    current.add(root.val);
    int sum = 0;

    for (int i = current.size() - 1; i >= 0; i--) {
      sum += current.get(i);
      if (sum == targetSum) {
        ans[0]++;
      }
    }

    hasPathSum3Helper(root.left, targetSum, new ArrayList<>(current), ans);
    hasPathSum3Helper(root.right, targetSum, new ArrayList<>(current), ans);
  }

  public static List<List<Integer>> pathSum2(TreeNode root, int targetSum) {
    List<List<Integer>> ans = new ArrayList<>();
    hasPathSum2Helper(root, targetSum, new ArrayList<>(), ans);

    return ans;
  }

  public static void hasPathSum2Helper(
      TreeNode root, int targetSum, List<Integer> current, List<List<Integer>> ans) {
    if (root == null)
      return;

    current.add(root.val);

    if (root.left == null && root.right == null && targetSum == root.val) {
      ans.add(new ArrayList<>(current));
    } else {
      hasPathSum2Helper(root.left, targetSum - root.val, current, ans);
      hasPathSum2Helper(root.right, targetSum - root.val, current, ans);
    }
    current.remove(current.size() - 1);
  }

  public boolean hasPathSum(TreeNode root, int targetSum) {
    if (root == null) {
      return false;
    }

    if (root.left == null && root.right == null) {
      return targetSum == root.val;
    }

    return hasPathSum(root.left, targetSum - root.val)
        || hasPathSum(root.right, targetSum - root.val);
  }

  public static List<List<Integer>> subsetsWithDup(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    Arrays.sort(nums);
    subsetsWithDupHelper(ans, new ArrayList<>(), nums, 0);

    return ans;
  }

  public static void subsetsWithDupHelper(
      List<List<Integer>> ans, List<Integer> current, int[] nums, int start) {
    ans.add(new ArrayList<>(current));

    for (int i = start; i < nums.length; i++) {
      if (i > start && nums[i] == nums[i - 1]) {
        continue;
      }
      current.add(nums[i]);
      subsetsWithDupHelper(ans, current, nums, i + 1);
      current.remove(current.size() - 1);
    }
  }

  public static List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    helperSubsets(ans, new ArrayList<>(), nums, 0);

    return ans;
  }

  public static void helperSubsets(
      List<List<Integer>> ans, List<Integer> current, int[] nums, int start) {
    if (nums.length == start) {
      ans.add(new ArrayList<>(current));
      return;
    }

    current.add(nums[start]);
    helperSubsets(ans, current, nums, start + 1);
    current.remove(current.size() - 1);
    helperSubsets(ans, current, nums, start + 1);
  }

  public static List<List<Integer>> combine(int n, int k) {
    List<List<Integer>> ans = new ArrayList<>();
    combineHelper(ans, new ArrayList<>(), n, k, 1);

    return ans;
  }

  public static void combineHelper(
      List<List<Integer>> ans, List<Integer> current, int n, int k, int number) {
    if (current.size() == k) {
      ans.add(new ArrayList<>(current));
      return;
    }

    for (int i = number; i <= n; i++) {
      current.add(i);
      combineHelper(ans, current, n, k, i + 1);
      current.remove(current.size() - 1);
    }
  }

  public static int findCenter(int[][] edges) {
    if (edges[0][0] == edges[1][0] || edges[0][0] == edges[1][1]) {
      return edges[0][0];
    } else {
      return edges[0][1];
    }
  }

  public static int buyChoco(int[] prices, int money) {
    Arrays.sort(prices);
    int cost = prices[0] + prices[1];

    if (cost <= money) {
      return money - cost;
    }

    return money;
  }

  public static boolean isValidBST(TreeNode root) {
    return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
  }

  public static boolean isValidBST(TreeNode root, long min, long max) {
    if (root == null)
      return true;
    if (root.val >= max || root.val <= min)
      return false;
    return isValidBST(root.left, min, root.val) && isValidBST(root.right, root.val, max);
  }

  public int maxProductDifference(int[] nums) {
    Arrays.sort(nums);

    return (nums[0] * nums[1]) - (nums[nums.length - 1] * nums[nums.length - 2]);
  }

  public static int jump2(int[] nums) {
    int steps = 0;
    int current = 0;
    int farthest = 0;

    for (int i = 0; i < nums.length - 1; i++) {
      farthest = Math.max(farthest, nums[i] + i);

      if (i == current) {
        steps += 1;
        current = farthest;
      }
    }

    return steps;
  }

  public static int maxArea(int[] height) {
    int maxWater = 0;
    int left = 0;
    int right = height.length - 1;

    while (left < right) {
      int minHeight = Math.min(height[left], height[right]);
      int width = right - left;
      int currentWater = minHeight * width;

      maxWater = Math.max(maxWater, currentWater);

      if (height[left] < height[right]) {
        left += 1;
      } else {
        right -= 1;
      }
    }

    return maxWater;
  }

  public static List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    Arrays.sort(nums);
    boolean[] used = new boolean[nums.length];
    permuteUniqueTraverse(nums, used, new ArrayList<>(), ans);

    return ans;
  }

  private static void permuteUniqueTraverse(
      int[] nums, boolean[] used, List<Integer> current, List<List<Integer>> ans) {
    if (current.size() == nums.length) {
      ans.add(new ArrayList<>(current));
      return;
    }

    for (int i = 0; i < nums.length; i++) {
      if (used[i] || (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])) {
        continue;
      }

      used[i] = true;
      current.add(nums[i]);
      permuteUniqueTraverse(nums, used, current, ans);
      current.remove(current.size() - 1);
      used[i] = false;
    }
  }

  public static List<String> letterCombinations(String digits) {
    List<String> ans = new ArrayList<>();
    if (digits == null || digits.length() == 0) {
      return ans;
    }

    Map<Character, String> map = new HashMap<>();
    map.put('2', "abc");
    map.put('3', "def");
    map.put('4', "ghi");
    map.put('5', "jkl");
    map.put('6', "mno");
    map.put('7', "pqrs");
    map.put('8', "tuv");
    map.put('9', "wxyz");

    backtrack(digits, 0, new StringBuilder(), map, ans);

    return ans;
  }

  private static void backtrack(
      String digits,
      int start,
      StringBuilder current,
      Map<Character, String> map,
      List<String> ans) {

    if (start == digits.length()) {
      ans.add(current.toString());
      return;
    }

    char digit = digits.charAt(start);
    String letters = map.get(digit);

    for (char letter : letters.toCharArray()) {
      current.append(letter);
      backtrack(digits, start + 1, current, map, ans);
      current.deleteCharAt(current.length() - 1);
    }
  }

  public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<List<Integer>> ans = new ArrayList<>();
    Arrays.sort(candidates);
    traverse2(candidates, target, 0, new ArrayList<>(), ans);

    return ans;
  }

  private static void traverse2(
      int[] candidates, int target, int start, List<Integer> current, List<List<Integer>> ans) {
    if (target == 0) {
      ans.add(new ArrayList<>(current));
      return;
    }

    for (int i = start; i < candidates.length; i++) {
      // skip dp
      if (i > start && candidates[i] == candidates[i - 1]) {
        continue;
      }

      // skip current target if candidate > remainining target
      if (candidates[i] > target) {
        break;
      }

      current.add(candidates[i]);
      traverse2(candidates, target - candidates[i], i + 1, current, ans);
      current.remove(current.size() - 1);
    }
  }

  public static List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> ans = new ArrayList<>();
    traverse(candidates, target, 0, new ArrayList<>(), ans);
    return ans;
  }

  public static void traverse(
      int[] candidates, int target, int start, List<Integer> combination, List<List<Integer>> ans) {

    int sum = 0;
    for (int num : combination) {
      sum += num;
    }

    if (sum == target) {
      ans.add(new ArrayList<>(combination));
      return;
    }

    if (sum > target)
      return;

    for (int i = start; i < candidates.length; i++) {
      combination.add(candidates[i]);
      traverse(candidates, target, i, combination, ans);
      combination.remove(combination.size() - 1);
    }
  }

  public static int numSubseq(int[] nums, int target) {
    return 0;
  }

  public int minNumber(int[] nums1, int[] nums2) {
    int min = Integer.MAX_VALUE;

    for (int j : nums1) {
      for (int k : nums2) {
        if (j == k) {
          min = Math.min(min, j);
        }

        min = Math.min(Math.min(j * 10 + k, k * 10 + j), min);
      }
    }

    return min;
  }

  public static List<List<String>> partition(String s) {
    List<List<String>> ans = new ArrayList<>();
    List<String> current = new ArrayList<>();

    partitionHelper(s, 0, current, ans);

    return ans;
  }

  private static void partitionHelper(
      String s, int i, List<String> current, List<List<String>> ans) {
    if (i == s.length()) {
      ans.add(new ArrayList<>(current));
      return;
    }

    for (int j = i + 1; j <= s.length(); j++) {
      String substring = s.substring(i, j);
      if (isPalindrome(substring)) {
        current.add(substring);
        partitionHelper(s, j, current, ans);
        current.remove(current.size() - 1);
      }
    }
  }

  public static int threeSumMulti(int[] arr, int target) {
    // TLE
    int res = 0;
    boolean found = false;
    int i = 0;

    while (i < arr.length && !found) {
      int j = i + 1;
      while (j < arr.length && !found) {
        int k = j + 1;
        while (k < arr.length && !found) {
          if ((arr[i] + arr[j] + arr[k]) == target) {
            res++;
          }
          k++;
        }
        j++;
      }
      i++;
    }

    return res;
  }

  public static int[] getNoZeroIntegers(int n) {
    // works just fine but is not accepted
    // int[] arr = new int[2];
    // for(int i = 0; i < n; i++){
    // int a = i;
    // int b = n - i;
    // if(a != 0 && a + b == n){
    // arr[0] = a;
    // arr[1] = b;
    // }
    // }
    //
    // return arr;

    for (int i = 1; i < n; i++) {
      if (nonZero(i) && nonZero(n - i)) {
        return new int[] { i, n - i };
      }
    }

    return new int[] { -1, -1 };
  }

  public static boolean nonZero(int n) {
    while (n > 0) {
      if (n % 10 == 0)
        return false;
      n /= 10;
    }
    return true;
  }

  public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
    return 0.0;
  }

  public static int minOperations(String s1, String s2, int x) {
    // 2896
    return 0;
  }

  public static int maxSumDivThree(int[] nums) {
    int[] dp = new int[3];
    for (int i : nums) {
      for (int j : Arrays.copyOf(dp, dp.length)) {
        dp[(i + j) % 3] = Math.max(dp[(i + j) % 3], (i + j));
      }
    }

    return dp[0];
  }

  public static int[][] onesMinusZeros(int[][] grid) {
    int[] row = new int[grid.length];
    int[] col = new int[grid[0].length];

    int[][] ans = new int[grid.length][grid[0].length];

    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[0].length; j++) {
        if (grid[i][j] == 0) {
          row[i]--;
          col[j]--;
        } else {
          row[i]++;
          col[j]++;
        }
      }
    }

    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[0].length; j++) {
        ans[i][j] = row[i] + col[j];
      }
    }

    return ans;
  }

  public int minimizeArrayValue(int[] nums) {
    // average value of subarray
    long answer = 0;
    long sum = 0;

    for (int i = 0; i < nums.length; i++) {
      sum += nums[i];
      answer = Math.max(answer, (sum + i) / (i + 1));
    }

    return (int) answer;
  }

  public static int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
      for (int c : coins) {
        if (i - c >= 0) {
          dp[i] = Math.min(dp[i], dp[i - c] + 1);
        }
      }
    }

    return dp[amount] != amount + 1 ? dp[amount] : -1;
  }

  public static int maximizeSum(int[] nums, int k) {
    int sum = 0;
    int biggestNumber = Integer.MIN_VALUE;
    for (int i = 0; i < nums.length; i++) {
      biggestNumber = Math.max(biggestNumber, nums[i]);
    }

    for (int i = 0; i < k; i++) {
      sum += biggestNumber;
      biggestNumber += 1;
    }

    return sum;
  }

  public static int concatenatedBinary(int n) {
    // doesnt work for rlly long n's
    int ans = 0;
    StringBuilder sb = new StringBuilder();
    for (int i = 1; i <= n; i++) {
      String numberInBinary = Integer.toBinaryString(i);
      sb.append(numberInBinary);
    }

    ans += Integer.parseInt(sb.toString());

    return Integer.parseUnsignedInt(String.valueOf(ans), 2);
  }

  public static List<Integer> findDisappearedNumbersx(int[] nums) {
    Set<Integer> hash = new HashSet<>();
    List<Integer> missing = new ArrayList<>();

    for (int num : nums) {
      hash.add(num);
    }

    for (int i = 0; i < nums.length; i++) {
      if (!hash.contains(i + 1)) {
        missing.add(i + 1);
      }
    }

    return missing;
  }

  public static boolean isPowerOfTwo(int n) {
    // return Math.ceil(Math.log(n) / Math.log(2)) == Math.floor(Math.log(n) /
    // Math.log(2));

    if (n == 0)
      return false;

    while (n != 1) {
      if (n % 2 != 0) {
        return false;
      }
      n /= 2;
    }

    return true;
  }

  public static int findSpecialInteger(int[] arr) {
    Map<Integer, Integer> map = new HashMap<>();
    int max = Integer.MIN_VALUE;
    int who = 0;

    for (int i = 0; i < arr.length; i++) {
      map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);
    }

    for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
      int freq = entry.getValue();
      if (freq > max) {
        max = freq;
        who = entry.getKey();
      }
    }

    return who;
  }

  public static long minimalKSum(int[] nums, int k) {
    // 2195
    return 1L;
  }

  public int peakIndexInMountainArray(int[] arr) {
    int left = 0;
    int right = arr.length - 1;

    while (left < right) {
      int pivot = left + (right - left) / 2;

      if (arr[pivot] < arr[pivot + 1]) {
        left = pivot + 1;
      } else {
        right = pivot;
      }
    }

    return left;
  }

  public static String restoreString(String s, int[] indices) {
    char[] ans = new char[s.length()];
    for (int i = 0; i < indices.length; i++) {
      ans[indices[i]] = s.charAt(i);
    }

    return new String(ans);
  }

  public static int minOperations(int[] nums, int x) {
    int totalSum = 0;
    for (int num : nums) {
      totalSum += num;
    }

    int target = totalSum - x;
    int currentSum = 0;
    int maxWindow = -1;

    int i = 0;
    for (int j = 0; j < nums.length; j++) {
      currentSum += nums[j];

      while (i <= j && currentSum > target) {
        currentSum -= nums[i];
        i++;
      }

      if (currentSum == target) {
        maxWindow = Math.max(maxWindow, j - i + 1);
      }
    }

    return maxWindow == -1 ? -1 : nums.length - maxWindow;
  }

  public static List<String> commonChars(String[] words) {
    List<String> result = new ArrayList<>();
    List<Map<Character, Integer>> resMap = new ArrayList<>();

    for (String word : words) {
      Map<Character, Integer> map = new HashMap<>();
      for (int i = 0; i < word.length(); i++) {
        map.put(word.charAt(i), map.getOrDefault(word.charAt(i), 0) + 1);
      }
      resMap.add(map);
    }

    for (char c : resMap.get(0).keySet()) {
      if (resMap.stream().allMatch(map -> map.containsKey(c))) {
        int minCount = resMap.stream().map(map -> map.get(c)).min(Integer::compareTo).orElse(0);

        for (int i = 0; i < minCount; i++) {
          result.add(String.valueOf(c));
        }
      }
    }

    return result;
  }

  public int[][] transpose(int[][] matrix) {
    // transpui matricea, schimb liniile cu coloanele

    int[][] result = new int[matrix[0].length][matrix.length];

    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; i++) {
        result[j][i] = matrix[i][j];
      }
    }

    return result;
  }

  public static int hammingDistance(int number1, int number2) {
    int x = number1 ^ number2;
    int setBits = 0;

    while (x > 0) {
      setBits += x & 1;
      x >>= 1;
    }

    return setBits;
  }

  public static boolean canWinNim(int n) {
    return n % 4 != 0;
  }

  public static int minimumAddedCoins(int[] coins, int target) {
    Arrays.sort(coins);

    int currentSum = 0;
    int coinsToAdd = 0;

    int index = 0;

    while (currentSum < target) {
      if (index < coins.length && coins[index] <= currentSum + 1) {
        currentSum += coins[index];
        index++;
      } else {
        currentSum += currentSum + 1;
        coinsToAdd++;
      }
    }

    return coinsToAdd;
  }

  public static int removeAlmostEqualCharacters(String word) {
    int n = word.length();

    int count = 0;
    int i = 1;

    while (i < n) {
      if (Math.abs(word.charAt(i) - word.charAt(i - 1)) <= 1) {
        count++;
        i += 2;
      } else {
        i++;
      }
    }

    return count;
  }

  public static int longestSubstring(String s, int k) {
    List<Character> evilLetters = new ArrayList<>();
    boolean goodSubstring = true;
    Map<Character, Integer> map = new HashMap<>();

    for (char letter : s.toCharArray()) {
      map.put(letter, map.getOrDefault(letter, 0) + 1);
    }

    for (char letter : map.keySet()) {
      if (map.get(letter) < k) {
        evilLetters.add(letter);
        goodSubstring = false;
      }
    }

    if (goodSubstring)
      return s.length();

    for (char letter : evilLetters) {
      s = s.replace(letter, ' ');
    }

    String[] afterDivide = s.split("\\s+");
    int ans = 0;

    for (String string : afterDivide) {
      ans = Math.max(ans, longestSubstring(string, k));
    }

    return ans;
  }

  public static int maxSubarrayLength(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<>();
    int maxLength = 0;

    for (int i = 0, j = 0; j < nums.length; j++) {
      map.put(nums[j], map.getOrDefault(nums[j], 0) + 1);

      while (map.get(nums[j]) > k) {
        map.put(nums[i], map.get(nums[i]) - 1);
        i++;
      }

      maxLength = Math.max(maxLength, j - i + 1);
    }

    return maxLength;
  }

  public static List<String> binaryTreePaths(TreeNode root) {
    List<String> res = new ArrayList<>();
    if (root != null) {
      dfs(root, "", res);
    }

    return res;
  }

  private static void dfs(TreeNode node, String path, List<String> result) {
    if (node.left == null && node.right == null) {
      result.add(path + node.val);
      return;
    }

    if (node.left != null) {
      dfs(node.left, path + node.val + "->", result);
    }

    if (node.right != null) {
      dfs(node.right, path + node.val + "->", result);
    }
  }

  public int titleToNumber(String columnTitle) {
    int ans = 0;

    for (int i = 0; i < columnTitle.length(); i++) {
      ans = ans * 26 + (columnTitle.charAt(i) - 'A' + 1);
    }

    return ans;
  }

  public String convertToTitle(int columnNumber) {
    StringBuilder ans = new StringBuilder();

    while (columnNumber > 0) {
      columnNumber--;

      ans.append((char) ((columnNumber) % 26 + 'A'));
      columnNumber = columnNumber / 26;
    }

    return ans.reverse().toString();
  }

  public static int minTimeToVisitAllPoints(int[][] points) {
    // Chebyshev
    int ans = 0;

    for (int i = 0; i < points.length - 1; i++) {
      int currX = points[i][0];
      int currY = points[i][1];

      int targetX = points[i + 1][0];
      int targetY = points[i + 1][1];

      ans += Math.max(Math.abs(targetX - currX), Math.abs(targetY - currY));
    }

    return ans;
  }

  public static int countCharacters(String[] words, String chars) {
    int sum = 0;

    Map<Character, Integer> charsCount = new HashMap<>();
    for (char c : chars.toCharArray()) {
      charsCount.put(c, charsCount.getOrDefault(c, 0) + 1);
    }

    for (String word : words) {
      Map<Character, Integer> wordCount = new HashMap<>();
      for (char c : word.toCharArray()) {
        wordCount.put(c, wordCount.getOrDefault(c, 0) + 1);
      }

      boolean good = true;

      for (Character c : wordCount.keySet()) {
        if (charsCount.getOrDefault(c, 0) < wordCount.get(c)) {
          good = false;
          break;
        }
      }

      if (good)
        sum += word.length();
    }

    return sum;
  }

  public static boolean arrayStringsAreEqual(String[] word1, String[] word2) {
    StringBuilder first = new StringBuilder();
    StringBuilder second = new StringBuilder();

    for (String words : word1) {
      first.append(words);
    }

    for (String words : word2) {
      second.append(words);
    }

    return first.compareTo(second) == 0;
  }

  public static int[] prisonAfterNDays(int[] cells, int n) {
    Map<String, Integer> seenStates = new HashMap<>();

    // Iterate until a cycle is detected or until n days are reached
    int i;
    for (i = 0; i < n; i++) {
      // Convert the current state to a string for checking repetition
      String currentStateStr = Arrays.toString(cells);

      // If the current state is already seen, a cycle is detected
      if (seenStates.containsKey(currentStateStr)) {
        // Calculate the length of the cycle
        int cycleLength = i - seenStates.get(currentStateStr);

        // Skip the remaining days to reach the final state
        int remainingDays = (n - i) % cycleLength;
        return prisonAfterNDays(cells, remainingDays);
      }

      // Store the current state and its day index
      seenStates.put(currentStateStr, i);

      // Update the cells for the next iteration
      cells = getNextDayState(cells);
    }

    // If we reach this point, there is no cycle, return the state after n days
    return cells;
  }

  private static int[] getNextDayState(int[] cells) {
    // TLE
    int[] nextDayCells = new int[cells.length];

    // Skip the first and last cells as they are always vacant
    for (int j = 1; j < cells.length - 1; j++) {
      // Check the neighbors to determine the state of the cell
      nextDayCells[j] = (cells[j - 1] == cells[j + 1]) ? 1 : 0;
    }

    return nextDayCells;
  }

  public static int trebuchet2() throws IOException {
    int sumInt = 0;
    String sum = "";

    Map<String, Integer> map = new HashMap<>();
    map.put("one", 1);
    map.put("two", 2);
    map.put("three", 3);
    map.put("four", 4);
    map.put("five", 5);
    map.put("six", 6);
    map.put("seven", 7);
    map.put("eight", 8);
    map.put("nine", 9);

    try {
      File file = new File(
          "C:\\Users\\tmdop\\AppData\\Roaming\\JetBrains\\IntelliJIdea2023.3\\scratches\\input.txt");
      Scanner scanner = new Scanner(file);

      while (scanner.hasNextLine()) {
        String line = scanner.nextLine();

        String wordsOnly = line.replaceAll("[^a-zA-Z]|(one|two|three|four|five|six|seven|eight|nine)", "$1");
        System.out.println(wordsOnly);
      }

      scanner.close();
    } catch (FileNotFoundException e) {
      System.out.println("Can't find the file");
      e.printStackTrace();
    }

    return sumInt;
  }

  private static int getDigitValue(String value, Map<String, Integer> map) {
    return map.containsKey(value.toLowerCase())
        ? map.get(value.toLowerCase())
        : Integer.parseInt(value);
  }

  public static int trebuchet() throws IOException {
    int sumInt = 0;
    String sum = "";
    try {
      File file = new File(
          "C:\\Users\\tmdop\\AppData\\Roaming\\JetBrains\\IntelliJIdea2023.3\\scratches\\input.txt");
      Scanner scanner = new Scanner(file);

      while (scanner.hasNextLine()) {
        String line = scanner.nextLine();

        String digits = line.replaceAll("[^0-9]", "");

        if (digits.length() > 1) {
          char first = digits.charAt(0);
          char last = digits.charAt(digits.length() - 1);

          sum = String.valueOf(first) + last;
          sumInt += Integer.parseInt(sum);

        } else if (digits.length() == 1) {
          char digit = digits.charAt(0);
          sum = String.valueOf(digit) + digit;
          sumInt += Integer.parseInt(sum);
        }
      }

      scanner.close();
    } catch (FileNotFoundException e) {
      System.out.println("Can't find the file");
      e.printStackTrace();
    }

    return sumInt;
  }

  public static int findTargetSumWays(int[] nums, int target) {
    return backtrack(nums, 0, 0, target);
  }

  public static int backtrack(int[] nums, int index, int currentSum, int target) {
    if (index == nums.length)
      return currentSum == target ? 1 : 0;

    int positive = backtrack(nums, index + 1, currentSum + nums[index], target);
    int negative = backtrack(nums, index + 1, currentSum - nums[index], target);

    return positive + negative;
  }

  public static String largestGoodInteger(String num) {
    String[] array = { "999", "888", "777", "666", "555", "444", "333", "222", "111", "000" };

    for (String number : array) {
      if (num.contains(number))
        return number;
    }

    return "";
  }

  public static String largestOddNumber(String num) {
    for (int i = num.length() - 1; i >= 0; i--) {
      int digit = num.charAt(i) - '0';

      if (digit % 2 != 0) {
        return num.substring(0, i + 1);
      }
    }

    return "";
  }

  public static int countSubstrings(String s, String t) {
    int count = 0;

    // generate substrings of s
    for (int i = 0; i < s.length(); i++) {
      for (int j = i + 1; j <= s.length(); j++) {
        String current = s.substring(i, j);

        // compare w substrings of t
        for (int m = 0; m < t.length(); m++) {
          for (int n = m + 1; n <= t.length(); n++) {
            String currentT = t.substring(m, n);

            if (differ(current, currentT))
              count++;
          }
        }
      }
    }

    return count;
  }

  public static boolean differ(String s, String t) {
    int differences = 0;
    int minLength = Math.min(s.length(), t.length());
    for (int i = 0; i < minLength; i++) {
      if (s.charAt(i) != t.charAt(i)) {
        differences++;
        if (differences > 1)
          return false;
      }
    }

    differences += Math.abs(s.length() - t.length());

    return differences == 1;
  }

  public static int numberOfSubstrings(String s) {
    int pointerA = -1;
    int pointerB = -1;
    int pointerC = -1;

    int count = 0;

    for (int i = 0; i < s.length(); i++) {
      if (s.charAt(i) == 'a') {
        pointerA = i;
      } else if (s.charAt(i) == 'b') {
        pointerB = i;
      } else {
        pointerC = i;
      }

      if (pointerA != -1 && pointerB != -1 && pointerC != -1) {
        count += Math.min(pointerA, Math.min(pointerC, pointerB)) + 1;
      }
    }

    return count;
  }

  public static int twoCitySchedCost(int[][] costs) {
    int sum = 0;
    int[] diff = new int[costs.length];
    for (int i = 0; i < costs.length; i++) {
      sum += costs[i][0];
      diff[i] = costs[i][1] - costs[i][0];
    }
    Arrays.sort(diff);
    for (int i = 0; i < costs.length / 2; i++) {
      sum += diff[i];
    }

    return sum;
  }

  public static int countNodes(TreeNode root) {
    if (root == null)
      return 0;

    int left = countNodes(root.left);
    int right = countNodes(root.right);

    return left + right + 1;
  }

  public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if (headA == null || headB == null) {
      return null;
    }

    ListNode originalA = headA;
    ListNode originalB = headB;

    while (originalA != originalB) {
      originalA = (originalA == null) ? headB : originalA.next;
      originalB = (originalB == null) ? headA : originalB.next;
    }

    return originalA;
  }

  // public static int triangularSum(int[] nums) {
  //
  // }
  public List<Integer> getRow(int rowIndex) {
    List<List<Integer>> triangle = generate(rowIndex + 1);
    return triangle.get(rowIndex);
  }

  public static List<List<Integer>> generate(int numRows) {
    List<List<Integer>> triangle = new ArrayList<>();

    if (numRows <= 0) {
      return triangle;
    }

    for (int i = 0; i < numRows; i++) {
      List<Integer> row = new ArrayList<>();

      for (int j = 0; j <= i; j++) {
        if (j == 0 || j == i) {
          row.add(1); // Values at the edges are always 1
        } else {
          int val = triangle.get(i - 1).get(j - 1) + triangle.get(i - 1).get(j);
          row.add(val);
        }
      }

      triangle.add(row);
    }

    return triangle;
  }

  public static boolean isMonotonic(int[] nums) {
    boolean increasing = true;
    boolean decreasing = true;
    for (int i = 1; i <= nums.length - 1; i++) {
      if (nums[i] > nums[i - 1]) {
        decreasing = false;
      } else if (nums[i] < nums[i - 1]) {
        increasing = false;
      }
    }

    return increasing || decreasing;
  }

  public static List<Integer> majorityElement(int[] nums) {
    int majorityElement = nums.length / 2;
    List<Integer> list = new ArrayList<>();
    for (int num : nums) {
      int count = 0;
      for (int i : nums) {
        if (num == i) {
          count++;
        }
      }
      if (count > majorityElement && !list.contains(num)) {
        list.add(num);
      }
    }
    return list;
  }

  public static String reversePrefix(String word, char ch) {
    if (!word.contains(Character.toString(ch))) {
      return word;
    }

    return reversePrefixWord(word, 0, word.indexOf(ch)) + word.substring(word.indexOf(ch) + 1);
  }

  public static String reversePrefixWord(String word, int start, int end) {
    StringBuilder sb = new StringBuilder();
    for (int i = end; i >= start; i--) {
      sb.append(word.charAt(i));
    }

    return sb.toString();
  }

  public static int numOfStrings(String[] patterns, String word) {
    int count = 0;
    for (String pattern : patterns) {
      if (word.contains(pattern))
        count++;
    }

    return count;
  }

  public static String[] sortPeople(String[] names, int[] heights) {
    Map<Integer, String> map = new HashMap<>();
    for (int i = 0; i < names.length; i++) {
      map.put(heights[i], names[i]);
    }

    Arrays.sort(heights);

    String[] ord = new String[names.length];
    int index = 0;
    for (int i = heights.length - 1; i >= 0; i--) {
      ord[index] = map.get(heights[i]);
      index++;
    }

    return ord;
  }

  public static List<String> summaryRanges(int[] nums) {
    List<String> list = new ArrayList<>();

    int start = 0;

    for (int i = 0; i < nums.length; i++) {
      if (i == nums.length - 1 || nums[i] + 1 != nums[i + 1]) {
        if (start == i) {
          list.add(nums[start] + "");
        } else {
          list.add(nums[start] + "->" + nums[i]);
        }
        start = i + 1;
      }
    }

    return list;
  }

  public static int alternateDigitSum(int n) {
    List<Integer> list = new ArrayList<>();
    int sum = 0;

    for (int i = 0; i <= n; i++) {
      list.add(n % 10);
      n /= 10;
    }

    sum += list.get(0);
    for (int i = list.size() - 1; i >= 1; i--) {
      if (i % 2 == 0) {
        sum += list.get(i);
      } else {
        sum -= list.get(i);
      }
    }

    return sum;
  }

  public static int[] separateDigits(int[] nums) {
    List<Integer> list = new ArrayList<>();
    for (int num : nums) {
      int copy = num;
      List<Integer> digits = new ArrayList<>();

      while (copy > 0) {
        digits.add(0, copy % 10);
        copy /= 10;
      }

      list.addAll(digits);
    }

    int[] ans = new int[list.size()];
    for (int i = 0; i < ans.length; i++) {
      ans[i] = list.get(i);
    }

    return ans;
  }

  public static int countEven(int num) {
    if (num == 0)
      return 0;

    int count = 0;

    for (int i = 1; i <= num; i++) {
      int copy = i;
      int sum = 0;

      while (copy > 0) {
        int digit = copy % 10;
        sum += digit;
        copy /= 10;
      }

      if (sum % 2 == 0)
        count++;
    }

    return count;
  }

  public static int getLucky(String s, int k) {
    StringBuilder sb = new StringBuilder();

    for (char c : s.toCharArray()) {
      int position = c - 'a' + 1;
      sb.append(position);
    }

    String current = sb.toString();

    for (int i = 0; i < k; i++) {
      int sum = 0;

      for (char digit : current.toCharArray()) {
        sum += digit - '0';
      }

      current = String.valueOf(sum);
    }

    return Integer.parseInt(current);
  }

  public static int addDigits(int num) {
    if (num == 0)
      return 0;
    int sum = 0;
    while (num > 0) {
      sum += num % 10;
      num /= 10;

      if (sum > 9 && num == 0) {
        num = sum;
        sum = 0;
      }
    }

    return sum;
  }

  public static int maxSum(int[] nums) {
    int sum;
    int max = 0;
    for (int i = 0; i < nums.length; i++) {
      sum = 0;
      for (int j = i + 1; j < nums.length; j++) {
        if (maxDigit(nums[i]) == maxDigit(nums[j]))
          sum = nums[i] + nums[j];
        if (sum > max)
          max = sum;
      }
    }
    if (max == 0) {
      return -1;
    } else {
      return max;
    }
  }

  public static int maxDigit(int num) {
    int largest = 0;
    while (num != 0) {
      largest = Math.max(largest, num % 10);
      num /= 10;
    }
    return largest;
  }

  public static boolean canBeEqual(String s1, String s2) {
    if (s1.length() != s2.length())
      return false;

    for (int i = 0; i < s1.length(); i++) {
      if (s1.charAt(i) != s2.charAt(i) && s1.charAt(i) != s2.charAt((i + 2) % s1.length()))
        return false;
    }
    return true;
  }

  public static int findPeakElement(int[] nums) {
    int left = 0;
    int right = nums.length - 1;

    while (left < right) {
      int mid = left + (right - left) / 2;

      if (nums[mid] < nums[mid + 1]) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    return left;
  }

  // public static boolean searchMatrix(int[][] matrix, int target) {
  // for (int[] ints : matrix) {
  // for (int j = 0; j < matrix.length; j++) {
  // if (ints[j] == target) return true;
  // }
  // }
  // return false;
  // }
  public static String gcdOfStrings(String str1, String str2) {
    if (!(str1 + str2).equals(str2 + str1))
      return "";

    int length1 = str1.length();
    int length2 = str2.length();

    int gcd = gcd(length1, length2);

    return str1.substring(0, gcd);
  }

  public static int gcd(int a, int b) {
    if (b == 0)
      return a;
    return gcd(b, a % b);
  }

  public static boolean repeatedSubstringPattern(String s) {
    String concat = s + s;
    return concat.substring(1, concat.length() - 1).contains(s);
  }

  public static int[] searchRange(int[] nums, int target) {
    // TLE
    int[] ans = new int[2];
    int right = 0;
    int left = nums.length - 1;
    int pivot;

    while (right <= left) {
      pivot = left + (right - left) / 2;

      if (nums[pivot] == target) {
        ans[0] = pivot;
        ans[1] = pivot + 1;
        return ans;
      } else {
        if (target < nums[pivot]) {
          right = pivot - 1;
        } else if (target > nums[pivot]) {
          left = pivot + 1;
        } else {
          ans[0] = -1;
          ans[1] = -1;
        }
      }
    }

    return ans;
  }

  public boolean isBalanced(TreeNode root) {
    if (root == null)
      return true;
    return Math.abs(height(root.left) - height(root.right)) <= 1
        && isBalanced(root.left)
        && isBalanced(root.right);
  }

  public int height(TreeNode currentNode) {
    if (currentNode == null)
      return 0;
    return 1 + Math.max(height(currentNode.left), height(currentNode.right));
  }

  public static String reverseVowels(String s) {
    char[] letters = s.toCharArray();
    int start = 0;
    int end = s.length() - 1;

    Set<Character> vowels = new HashSet<>();
    vowels.add('a');
    vowels.add('e');
    vowels.add('i');
    vowels.add('o');
    vowels.add('u');
    vowels.add('A');
    vowels.add('E');
    vowels.add('I');
    vowels.add('O');
    vowels.add('U');

    while (start < end) {
      while (start < end && !vowels.contains(letters[start])) {
        start++;
      }

      while (start < end && !vowels.contains(letters[end])) {
        end--;
      }

      char temp = letters[start];
      letters[start] = letters[end];
      letters[end] = temp;

      start++;
      end--;
    }

    return new String(letters);
  }

  public static int tribonacci(int n) {
    int[] array = new int[38];
    array[0] = 0;
    array[1] = array[2] = 1;

    for (int i = 3; i <= n; i++) {
      array[i] = array[i - 3] + array[i - 2] + array[i - 1];
    }

    return array[n];
  }

  public static boolean checkArray(int[] nums, int k) {
    return false;
  }

  public static boolean checkSubarraySum(int[] nums, int k) {
    if (nums == null || nums.length < 2)
      return false;

    // Handle the special case where k is 0
    if (k == 0) {
      for (int i = 0; i < nums.length - 1; i++) {
        if (nums[i] == 0 && nums[i + 1] == 0) {
          return true;
        }
      }
      return false;
    }

    int sum = 0;
    Map<Integer, Integer> remainderMap = new HashMap<>();
    remainderMap.put(0, -1); // Initialize with a remainder of 0 at index -1

    for (int i = 0; i < nums.length; i++) {
      sum += nums[i];
      sum %= k;

      if (remainderMap.containsKey(sum)) {
        if (i - remainderMap.get(sum) >= 2) {
          return true;
        }
      } else {
        remainderMap.put(sum, i);
      }
    }

    return false;
    // TLE
    // if(k == 0){
    // for(int i = 0; i < nums.length - 1; i++){
    // if(nums[i] == 0 && nums[i + 1] == 0) return true;
    // }
    // return false;
    // }
    //
    // for(int i = 0; i < nums.length; i++){
    // int sum = 0;
    // for(int j = i; j < nums.length; j++){
    // sum += nums[j];
    // if(sum % k == 0 && j - i + 1 >= 2) return true;
    // }
    // }
    // return false;
    // }
  }

  public static int maxProduct(int[] nums) {
    int max = Integer.MIN_VALUE;
    for (int i = 0; i < nums.length; i++) {
      int product = 1;
      for (int j = i; j < nums.length; j++) {
        product *= nums[j];
        if (product > max)
          max = product;
      }
    }
    return max;

    // works for 111/190
    // int max = Integer.MIN_VALUE;
    // int product = 1;
    // int j = 0;
    //
    // for(int i = 0; i < nums.length; i++){
    // product *= nums[i];
    // if(product > max) max = product;
    // }
    // return max;
  }

  public static int numSubarrayProductLessThanK(int[] nums, int k) {
    // this passes everything
    if (k <= 1)
      return 0;

    int count = 0;
    int product = 1;
    int j = 0;

    for (int i = 0; i < nums.length; i++) {
      product *= nums[i];
      while (product >= k) {
        product /= nums[j];
        j++;
      }
      count += (i - j + 1);
    }

    return count;

    // this is fine but passes 39/98 on leetcode
    // if(k <= 1) return 0;
    // int count = 0;
    // for(int i = 0; i < nums.length; i++){
    // int product = 1;
    // for(int j = i; j < nums.length; j++){
    // product *= nums[j];
    // if(product < k) count++;
    // }
    // }
    // return count;
  }

  public static int subarraySum(int[] nums, int k) {
    int count = 0;
    for (int i = 0; i < nums.length; i++) {
      int sum = 0;
      for (int j = i; j < nums.length; j++) {
        sum += nums[j];
        if (sum == k)
          count++;
      }
    }
    return count;
  }

  public static List<List<Integer>> minimumAbsDifference(int[] arr) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(arr);

    int min = Integer.MAX_VALUE;

    for (int i = 0; i + 1 < arr.length; i++) {
      if (arr[i + 1] - arr[i] < min) {
        min = arr[i + 1] - arr[i];
        list.clear();
      }

      if (arr[i + 1] - arr[i] == min) {
        list.add(Arrays.asList(arr[i], arr[i + 1]));
      }
    }

    return list;
  }

  public static boolean checkPowersOfThree(int n) {
    while (n > 0) {
      if (n % 3 == 2)
        return false;
      n /= 3;
    }
    return true;
  }

  public static int numDifferentIntegers(String word) {
    // EXTRACT NUMBERS FROM STRING
    String[] split = word.split("[a-z]+");
    Set<String> set = new HashSet<>();
    for (String s : split) {
      if (!s.isEmpty()) {
        set.add(s.replaceFirst("^0+(?!$)", ""));
      }
    }
    return set.size();
  }

  public static boolean check(int[] nums) {
    int count = 0;
    for (int i = 0; i < nums.length; i++) {
      if (nums[i] > nums[(i + 1) % nums.length])
        count++;
    }
    return count <= 1;
  }

  public static int countBalls(int lowLimit, int highLimit) {
    int[] boxes = new int[highLimit - lowLimit + 1];
    for (int i = lowLimit; i <= highLimit; i++) {
      int sum = 0;
      int copy = i;
      while (copy > 0) {
        sum += copy % 10;
        copy /= 10;
      }
      boxes[sum]++;
    }
    return Arrays.stream(boxes).max().getAsInt();
  }

  public static int numberOfEmployeesWhoMetTarget(int[] hours, int target) {
    int res = 0;
    for (int hour : hours) {
      if (hour >= target)
        res++;
    }

    return res;
  }

  // public static ListNode partition(ListNode head, int x) {
  // }

  public static int findNumberOfLIS(int[] nums) {
    // idk lmao
    return 0;
  }

  // public static int maximumBeauty(int[] nums, int k) {
  // Arrays.sort(nums);
  // int max = 0;
  // int left = 0;
  // int right = 0;
  //
  // while(right < nums.length){
  // if(nums[left] + k >= nums[right] - k){
  // max = Math.max(max, nums[right] - nums[left]);
  // right++;
  // } else {
  // left++;
  // }
  // }
  // return max;
  // }
  public static List<Integer> rightSideView(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    if (root == null)
      return result;

    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while (!queue.isEmpty()) {
      int size = queue.size();
      for (int i = 0; i < size; i++) {
        TreeNode current = queue.remove();
        if (i == size - 1) {
          result.add(current.val);
        }
        if (current.left != null)
          queue.add(current.left);
        if (current.right != null)
          queue.add(current.right);
      }
    }
    return result;
  }

  public static String strWithout3a3b(int a, int b) {
    StringBuilder s = new StringBuilder(a + b);
    while (a + b > 0) {
      String snack = s.toString();
      if (snack.endsWith("aa")) {
        s.append('b');
        b--;
      } else if (snack.endsWith("bb")) {
        s.append('a');
        a--;
      } else if (a >= b) {
        s.append('a');
        a--;
      } else {
        s.append('b');
        b--;
      }
    }
    return s.toString();
  }

  public static List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    List<Integer> current = new ArrayList<>();
    permuteHelper(nums, current, result);
    return result;
  }

  public static void permuteHelper(int[] nums, List<Integer> current, List<List<Integer>> result) {
    if (current.size() == nums.length) {
      result.add(new ArrayList<>(current));
      return;
    }

    for (int num : nums) {
      if (!current.contains(num)) {
        current.add(num);
        permuteHelper(nums, current, result);
        current.remove(current.size() - 1);
      }
    }
  }

  public static void sortColors(int[] nums) {
    int i;
    int count0 = 0;
    int count1 = 0;
    int count2 = 0;

    for (i = 0; i < nums.length; i++) {
      switch (nums[i]) {
        case 0:
          count0++;
          break;
        case 1:
          count1++;
          break;
        case 2:
          count2++;
          break;
      }
    }

    i = 0;

    while (count0 > 0) {
      nums[i++] = 0;
      count0--;
    }

    while (count1 > 0) {
      nums[i++] = 1;
      count1--;
    }

    while (count2 > 0) {
      nums[i++] = 2;
      count2--;
    }

    System.out.println(Arrays.toString(nums));
  }

  public static List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> map = new HashMap<>();
    for (String s : strs) {
      char[] chars = s.toCharArray();
      Arrays.sort(chars);
      String sorted = new String(chars);
      if (!map.containsKey(sorted)) {
        map.put(sorted, new ArrayList<>());
      }
      map.get(sorted).add(s);
    }
    return new ArrayList<>(map.values());
  }

  // public int[] evenOddBit(int n) {
  // String string = new
  // StringBuilder(Integer.toBinaryString(n)).reverse().toString();
  // int even = 0;
  // int odd = 0;
  // for(int i = 0; i < string.length(); i++){
  // if(string.charAt(i) == '1'){
  // if (i % 2 == 0) even++;
  // else odd++;
  //
  // }
  //
  // }
  // return new int[]{even, odd};
  // }
  public static int[] evenOddBit(int n) {
    return new int[] { Integer.bitCount(n & 0b0101010101), Integer.bitCount(n & 0b1010101010) };
    /*
     * n & 0b0101010101:
     * The binary number 0b0101010101 is a 10-bit number (0-indexed) that has ones
     * at even indices (e.g., 0, 2, 4, etc.) and zeros at odd indices (e.g., 1, 3,
     * 5, etc.).
     * The & (bitwise AND) operation is used to retain only the bits from n that
     * correspond to even indices.
     * 
     * n & 0b1010101010:
     * The binary number 0b1010101010 is also a 10-bit number (0-indexed) that has
     * zeros at even indices and ones at odd indices.
     * The & (bitwise AND) operation is used to retain only the bits from n that
     * correspond to odd indices.
     * 
     * Integer.bitCount(x): This method is used to count the number of ones (set
     * bits) in the binary representation of the integer x
     */
  }

  public static int missingNumber(int[] nums) {
    int n = nums.length;
    int sum = 0;
    for (int num : nums) {
      sum += num;
    }
    return (n * (n + 1) / 2) - sum;
  }

  public static void nextPermutation(int[] nums) {
    // smallest from right
    int i = nums.length - 2;
    while (i >= 0 && nums[i] >= nums[i + 1]) {
      i--;
    }

    // find the first element greater than nums[i] from the right
    if (i >= 0) {
      int j = nums.length - 1;
      while (j >= 0 && nums[j] <= nums[i]) {
        j--;
      }
      // swap nums[i] and nums[j]
      swap(nums, i, j);
    }
    Arrays.sort(nums, i + 1, nums.length);
  }

  private static void swap(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
  }

  public static int orangesRotting(int[][] grid) {
    int minutes = 0;
    int freshOranges = 0;
    Queue<int[]> rottenOranges = new LinkedList<>();

    // Add all rotten oranges to the queue
    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[0].length; j++) {
        // If the orange is rotten, add it to the queue
        if (grid[i][j] == 2) {
          rottenOranges.add(new int[] { i, j });
        } else if (grid[i][j] == 1) {
          freshOranges++;
        }
      }
    }

    int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

    // bfs to rot the oranges
    while (!rottenOranges.isEmpty() && freshOranges > 0) {
      int size = rottenOranges.size();

      for (int i = 0; i < size; i++) {
        int[] rotten = rottenOranges.poll();
        int x = rotten[0];
        int y = rotten[1];

        for (int[] direction : directions) {
          int newX = x + direction[0];
          int newY = y + direction[1];

          // actual rotting
          // if it fits in range and has a fresh orange
          if (newX >= 0
              && newX < grid.length
              && newY >= 0
              && newY < grid[0].length
              && grid[newX][newY] == 1) {
            grid[newX][newY] = 2;
            rottenOranges.add(new int[] { newX, newY });
            freshOranges--;
          }
        }
      }
      if (!rottenOranges.isEmpty())
        minutes++;
    }
    return freshOranges == 0 ? minutes : -1;
  }

  public static boolean isValidSudoku(char[][] board) {
    // Check rows
    for (int i = 0; i < 9; i++) {
      if (!isValidRow(board, i))
        return false;
    }

    // Check columns
    for (int i = 0; i < 9; i++) {
      if (!isValidColumn(board, i))
        return false;
    }

    // Check 3x3 boxes
    for (int i = 0; i < 9; i += 3) {
      for (int j = 0; j < 9; j += 3) {
        if (!isValidBox(board, i, j))
          return false;
      }
    }

    return true;
  }

  public static boolean isValidRow(char[][] board, int row) {
    Set<Character> set = new HashSet<>();
    for (int i = 0; i < 9; i++) {
      if (set.contains(board[row][i]))
        return false;
      if (board[row][i] != '.')
        set.add(board[row][i]);
    }
    return true;
  }

  public static boolean isValidColumn(char[][] board, int col) {
    Set<Character> set = new HashSet<>();
    for (int i = 0; i < 9; i++) {
      if (set.contains(board[i][col]))
        return false;
      if (board[i][col] != '.')
        set.add(board[i][col]);
    }
    return true;
  }

  public static boolean isValidBox(char[][] board, int row, int col) {
    Set<Character> set = new HashSet<>();
    for (int i = row; i < row + 3; i++) {
      for (int j = col; j < col + 3; j++) {
        if (set.contains(board[i][j]))
          return false;
        if (board[i][j] != '.')
          set.add(board[i][j]);
      }
    }
    return true;
  }

  public static int findComplement(int num) {
    StringBuilder sb = new StringBuilder();
    sb.append(Integer.toBinaryString(num));
    for (int i = 0; i < sb.length(); i++) {
      if (sb.charAt(i) == '0')
        sb.setCharAt(i, '1');
      else
        sb.setCharAt(i, '0');
    }
    return Integer.parseInt(sb.toString(), 2);
  }

  public boolean isUgly(int n) {
    if (n == 0)
      return false;
    while (n % 2 == 0)
      n /= 2;
    while (n % 3 == 0)
      n /= 3;
    while (n % 5 == 0)
      n /= 5;
    if (n == 1)
      return true;
    else
      return false;
  }

  public static int sumOfLeftLeaves(TreeNode root) {
    if (root == null)
      return 0;
    int sum = 0;
    if (root.left != null && root.left.left == null && root.left.right == null)
      sum += root.left.val;
    return sum + sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
  }

  public static boolean isHappy(int n) {
    int copy = n;
    int sum = 0;
    while (copy > 0) {
      sum += Math.pow(copy % 10, 2);
      copy /= 10;
    }
    if (sum == 1)
      return true;
    else if (sum == 4)
      return false;
    else
      return isHappy(sum);
  }

  public String removeOccurrences(String s, String part) {
    while (s.contains(part))
      s = s.replace(part, "");
    return s;
  }

  public static boolean isPrefixString(String s, String[] words) {
    StringBuilder sb = new StringBuilder();
    for (String word : words) {
      sb.append(word);
      if (sb.toString().equals(s))
        return true;
    }
    return false;
  }

  public static int longestConsecutive(int[] nums) {
    // bonus e si O(n) time complexity
    if (nums.length == 0)
      return 0;
    int longestConsecutive = 0;
    int currentConsecutive = 0;
    Arrays.sort(nums);
    for (int i = 0; i < nums.length - 1; i++) {
      if (nums[i + 1] - nums[i] == 1) {
        currentConsecutive++;
      } else if (nums[i + 1] - nums[i] > 1) {
        currentConsecutive = 0;
      }
      if (currentConsecutive > longestConsecutive) {
        longestConsecutive = currentConsecutive;
      }
    }
    return longestConsecutive + 1;
  }

  public static int vowelStrings(String[] words, int left, int right) {
    int countVowelStrings = 0;
    for (int i = left; i <= right; i++) {
      if (isVowelString(words[i])) {
        countVowelStrings++;
      }
    }
    return countVowelStrings;
  }

  public static boolean isVowelString(String s) {
    for (int i = 0; i <= s.length() - 1; i++) {
      if ((s.charAt(0) == 'a'
          || s.charAt(0) == 'e'
          || s.charAt(0) == 'i'
          || s.charAt(0) == 'o'
          || s.charAt(0) == 'u')
          && (s.charAt(s.length() - 1) == 'a'
              || s.charAt(s.length() - 1) == 'e'
              || s.charAt(s.length() - 1) == 'i'
              || s.charAt(s.length() - 1) == 'o'
              || s.charAt(s.length() - 1) == 'u')) {
        return true;
      }
    }
    return false;
  }

  public static int countGoodSubstrings(String s) {
    int count = 0;
    for (int i = 0; i < s.length() - 2; i++) {
      String sub = s.substring(i, i + 3);
      if (sub.charAt(0) != sub.charAt(1)
          && sub.charAt(1) != sub.charAt(2)
          && sub.charAt(0) != sub.charAt(2)) {
        count++;
      }
    }
    return count;
  }

  public static String longestNiceSubstring(String s) {
    String answer = "";
    for (int i = 0; i < s.length(); i++) {
      for (int j = i + 1; j <= s.length(); j++) {
        String sub = s.substring(i, j);
        if (isNice(sub) && sub.length() > answer.length()) {
          answer = sub;
        }
      }
    }
    return answer;
  }

  public static boolean isNice(String s) {
    Set<Character> uppercaseChars = new HashSet<>();
    Set<Character> lowercaseChars = new HashSet<>();
    for (char c : s.toCharArray()) {
      if (Character.isUpperCase(c)) {
        uppercaseChars.add(c);
      } else if (Character.isLowerCase(c)) {
        lowercaseChars.add(c);
      }
    }
    for (char c : s.toCharArray()) {
      if (Character.isUpperCase(c) && !lowercaseChars.contains(Character.toLowerCase(c))) {
        return false;
      }
      if (Character.isLowerCase(c) && !uppercaseChars.contains(Character.toUpperCase(c))) {
        return false;
      }
    }
    return true;
  }

  public static double findMaxAverage(int[] nums, int k) {
    double currentSum = 0;
    for (int i = 0; i < k; i++) {
      currentSum += nums[i];
    }

    double answer = currentSum / k;

    // going through subarrays
    for (int i = k; i < nums.length; i++) {
      // average of each subarray
      currentSum += nums[i] - nums[i - k];
      // biggest average
      answer = Math.max(answer, currentSum / k);
    }

    return answer;
  }

  public static int findMaxSum(int[] arr, int k) {
    int start;
    int end;
    int maxSum = 0;
    int currentSum;
    for (start = 0; start < arr.length - k; start++) {
      currentSum = 0;

      // sum of subarray elements
      for (end = start; end < start + k; end++) {
        currentSum += arr[end];
      }
      if (currentSum > maxSum) {
        maxSum = currentSum;
      }
    }
    return maxSum;
  }

  public static int divisorSubstrings(int num, int k) {
    int count = 0;
    String numString = Integer.toString(num);

    for (int i = 0; i < numString.length() - k + 1; i++) {
      int curr = Integer.parseInt(numString.substring(i, i + k));
      if (curr != 0 && num % curr == 0) {
        count++;
      }
    }

    return count;
  }

  // public static List<List<Integer>> threeSum(int[] nums) {
  // // doesnt handle duplicates
  // List<List<Integer>> ans = new ArrayList<>();
  // int i = 0;
  // while (i < nums.length - 2) {
  // int j = i + 1;
  // while (j < nums.length - 1) {
  // int k = j + 1;
  // while (k < nums.length) {
  // if (nums[i] + nums[j] + nums[k] == 0) {
  // List<Integer> triplet = Arrays.asList(nums[i], nums[j], nums[k]);
  // if (!ans.contains(triplet)) {
  // ans.add(triplet);
  // }
  // }
  // k++;
  // }
  // j++;
  // }
  // i++;
  // }
  // return ans;
  // }

  public static String addStrings(String num1, String num2) {
    long n1 = Long.parseLong(num1);
    long n2 = Long.parseLong(num2);
    return String.valueOf(n1 + n2);
  }

  public static int thirdMax(int[] nums) {
    Arrays.sort(nums);
    int count = 1;
    for (int i = nums.length - 1; i > 0; i--) {
      if (nums[i] != nums[i - 1])
        count++;
      if (count == 3)
        return nums[i - 1];
    }
    return nums[nums.length - 1];
  }

  public static List<String> fizzBuzz(int n) {
    List<String> list = new ArrayList<>();
    for (int i = 1; i <= n; i++) {
      if (i % 15 == 0)
        list.add("FizzBuzz");
      else if (i % 3 == 0)
        list.add("Fizz");
      else if (i % 5 == 0)
        list.add("Buzz");
      else
        list.add(String.valueOf(i));
    }
    return list;
  }

  public static int firstUniqChar(String s) {
    char[] letters = s.toCharArray();
    Map<Character, Integer> map = new HashMap<>();
    for (char c : letters) {
      map.put(c, map.getOrDefault(c, 0) + 1);
    }
    for (int i = 0; i < letters.length; i++) {
      if (map.get(letters[i]) == 1)
        return i;
    }
    return -1;
  }

  public static char findTheDifference(String s, String t) {
    char[] sArr = s.toCharArray();
    char[] tArr = t.toCharArray();
    Arrays.sort(sArr);
    Arrays.sort(tArr);
    for (int i = 0; i < sArr.length; i++) {
      if (sArr[i] != tArr[i])
        return tArr[i];
    }
    return tArr[tArr.length - 1];
  }

  public static String toHex(int num) {
    return Integer.toHexString(num);
  }

  public static int myAtoi(String s) {
    s = s.trim();
    if (s.length() == 0)
      return 0;
    if (s.length() == 1 && !Character.isDigit(s.charAt(0)))
      return 0;
    for (int i = 0; i < s.length(); i++) {
      if (!Character.isDigit(s.charAt(i)) && s.charAt(i) != '-' && s.charAt(i) != '+') {
        s = s.substring(0, i);
        break;
      }
    }

    return Integer.parseInt(s);
  }

  public static int[] topKFrequent(int[] nums, int k) {
    // Set<Integer> set = new HashSet<>();
    // int[] result = new int[k];
    // for (int j : nums) {
    // set.add(j);
    // }
    // for(int i = 0; i < result.length; i++){
    // result[i] = set.iterator().next();
    // set.remove(result[i]);
    // }
    // return result; WORKS FOR SOME CASES
    Map<Integer, Integer> map = new HashMap<>();
    for (int i : nums) {
      map.put(i, map.getOrDefault(i, 0) + 1);
    }
    List<Integer> list = new ArrayList<>(map.keySet());

    // (a, b) represents two elements (numbers) that are being compared in the
    // sorting process.
    // map.get(b) retrieves the frequency (count) of element b from the map.
    // map.get(a) retrieves the frequency (count) of element a from the map.
    // The comparator subtracts the frequency of a from the frequency of b. Here's
    // what happens when
    // we compare two elements:
    //
    // If the result of map.get(b) - map.get(a) is negative, it means that the
    // frequency of b is
    // less than the frequency of a. So b should come before a in the sorted list.
    // If the result is zero, it means that the frequencies of a and b are the same.
    // In this case,
    // the original order of the elements is maintained.
    // If the result is positive, it means that the frequency of b is greater than
    // the frequency of
    // a. So b should come after a in the sorted list.
    list.sort((a, b) -> map.get(b) - map.get(a));

    int[] result = new int[k];
    for (int i = 0; i < k; i++) {
      result[i] = list.get(i);
    }
    return result;
  }

  public static int findDuplicate(int[] nums) {
    Arrays.sort(nums);
    for (int i = 0; i < nums.length - 1; i++) {
      if (nums[i] == nums[i + 1])
        return nums[i];
    }
    return -1;
  }

  public static double[] convertTemperature(double celsius) {
    double[] result = new double[2];
    result[0] = celsius * 9 / 5 + 32;
    result[1] = celsius + 273.15;
    return result;
  }

  public static int candy(int[] ratings) {
    // int[] candies = new int[ratings.length];
    // Arrays.fill(candies, 1);
    // for(int i = 1; i < ratings.length; i++){
    // if(ratings[i] > ratings[i - 1]){
    // candies[i] = candies[i - 1] + 1;
    // }
    // }
    // for(int i = ratings.length - 2; i >= 0; i--){
    // if(ratings[i] > ratings[i + 1]){
    // candies[i] = Math.max(candies[i], candies[i + 1] + 1);
    // }
    // }
    // int sum = 0;
    // for (int candy : candies) {
    // sum += candy;
    // }
    // return sum;
    return 0;
  }

  public static int canCompleteCircuit(int[] gas, int[] cost) {
    int tank = 0;
    int start = 0;
    int total = 0;
    for (int i = 0; i < gas.length; i++) {
      tank += gas[i] - cost[i];
      if (tank < 0) {
        start = i + 1;
        tank = 0;
      }
      total += gas[i] - cost[i];
    }
    return total < 0 ? -1 : start % gas.length;
  }

  public int[] productExceptSelfBRUTEFORCE(int[] nums) {
    int end = nums.length;
    int[] result = new int[end];

    for (int i = 0; i < end; i++) {
      int one = 1;
      for (int j = 0; j < end; j++) {
        if (i != j) {
          one *= nums[j];
        }
      }
      result[i] = one;
    }
    return result;
  }

  public int hIndex(int[] citations) {
    Arrays.sort(citations);
    int n = citations.length;
    for (int i = 0; i < n; i++) {
      if (citations[i] >= n - i) {
        return n - i;
      }
    }
    return 0;
  }

  public static boolean canJump(int[] nums) {
    int max = 0;
    for (int i = 0; i < nums.length && i <= max; i++) {
      max = Math.max(max, 1 + nums[i]);
    }
    return max >= nums.length - 1;
  }

  public static int maxProfit2(int[] prices) {
    int maxProfit = 0;
    int end = prices.length;
    for (int i = 1; i < end; i++) {
      if (prices[i] > prices[i - 1]) {
        maxProfit += prices[i] - prices[i - 1];
      }
    }
    return maxProfit;
  }

  public static int removeDuplicates(int[] nums) {
    int frequency;
    int k = 0;
    for (int i = 0; i < nums.length; i++) {
      frequency = 0;
      for (int j = i + 1; j < nums.length; j++) {
        if (nums[i] == nums[j])
          frequency++;
      }
      if (frequency <= 2) {
        nums[k] = nums[i];
        k++;
      }
    }
    for (int i = 0; i < k; i++) {
    }

    return k;
  }

  public static boolean containsNearbyDuplicate(int[] nums, int k) {
    for (int i = 0; i < nums.length; i++) {
      for (int j = i + 1; j < nums.length; j++) {
        if (nums[i] == nums[j] && Math.abs(i - j) <= k) {
          return true;
        }
      }
    }
    return false;
  }

  public static int mostFrequentEven(int[] nums) {
    // even = par
    // odd = impar
    int[] arr = new int[100001];
    for (int i : nums) {
      if (i % 2 == 0)
        arr[i]++;
    }
    int max = -1, ans = 0;
    for (int i = 0; i < 100001; i++) {
      if (arr[i] > max) {
        max = arr[i];
        ans = i;
      }
    }
    if (max == 0)
      return -1;
    return ans;
  }

  public static int majorityElementOLD(int[] nums) {
    int majorityElement = nums.length / 2;
    for (int num : nums) {
      int count = 0;
      for (int i : nums) {
        if (num == i) {
          count++;
        }
      }
      if (count > majorityElement) {
        return num;
      }
    }
    return -1;
  }

  public static boolean wordPattern(String pattern, String s) {
    // TODO: Implement
    String[] words = s.split(" ");
    if (words.length != pattern.length()) {
      return false;
    }

    Map<String, Integer> map = new HashMap<>();
    for (int i = 0; i < words.length; i++) {
      map.put(words[i], map.getOrDefault(words[i], 0) + 1);
      System.out.println(words[i]);
    }
    return true;
  }

  public TreeNode invertTree(TreeNode root) {
    if (root == null) {
      return null;
    }

    TreeNode temp = root.left;
    root.left = root.right;
    root.right = temp;
    invertTree(root.left);
    invertTree(root.right);
    return root;
  }

  public static List<Integer> findDisappearedNumbers(int[] nums) {
    // //TLE for huge input
    // List<Integer> second = new ArrayList<>(nums.length);
    // for (int i = 1; i <= nums.length; i++) {
    // second.add(i);
    // }
    // for (int i = 0; i < nums.length; i++) {
    // if(second.contains(nums[i])) {
    // second.remove(Integer.valueOf(nums[i]));
    // }
    // }
    // return second;
    ArrayList<Integer> list = new ArrayList<>();
    HashSet<Integer> hash = new HashSet<>();

    for (int num : nums) {
      hash.add(num);
    }
    for (int i = 0; i < nums.length; i++) {
      if (!hash.contains(i + 1)) {
        list.add(i + 1);
      }
    }
    return list;
  }

  public int heightChecker(int[] heights) {
    int[] sortedHeights = Arrays.copyOf(heights, heights.length);
    Arrays.sort(sortedHeights);
    int count = 0;
    for (int i = 0; i < heights.length; i++) {
      if (heights[i] != sortedHeights[i]) {
        count++;
      }
    }
    return count;
  }

  public static int[] sortArrayByParity(int[] nums) {
    int[] res = new int[nums.length];
    int head = 0;
    int tail = nums.length - 1;
    for (int num : nums) {
      if (num % 2 == 0) {
        res[head] = num;
        head++;
      } else {
        res[tail] = num;
        tail--;
      }
    }
    return res;
  }

  public boolean checkIfExist(int[] arr) {
    for (int i = 0; i < arr.length; i++) {
      for (int j = 0; j < arr.length; j++) {
        if (i != j && arr[i] == arr[j] * 2) {
          return true;
        }
      }
    }
    return false;
  }

  public static int removeElements(int[] nums, int val) {
    // return IntStream.range(0, nums.length).filter(i -> nums[i] != val).map(i ->
    // nums[i]).toArray().length;
    // remove elements in place 2 POINTERS APPROACH
    int i = 0;
    for (int j = 0; j < nums.length; j++) {
      if (nums[j] != val) {
        nums[i] = nums[j];
        i++;
      }
    }
    return i;

    // public int removeDuplicates(int[] nums) {
    // int i = 0;
    // for (int j = 1; j < nums.length; j++) {
    // if (nums[i] != nums[j]) {
    // i++;
    // nums[i] = nums[j];
    // }
    // }
    // return i + 1;
    // }
  }

  public static void duplicateZeros(int[] arr) {

    Queue<Integer> q = new LinkedList<>();

    for (int i = 0; i < arr.length; i++) {
      q.add(arr[i]);

      if (arr[i] == 0)
        q.add(0);

      arr[i] = q.remove();
      System.out.println("we just added this to array " + arr[i]);
    }
  }

  public static int findNumbers(int[] nums) {
    int count = 0;
    for (int num : nums) {
      if (String.valueOf(num).length() % 2 == 0) {
        count++;
      }
    }
    return count;
  }

  public static int findMaxConsecutiveOnes(int[] nums) {
    int max = 0;
    for (int i = 0; i < nums.length; i++) {
      int count = 0;
      for (int j = i; j < nums.length; j++) {
        if (nums[j] == 1) {
          count++;
        } else {
          break;
        }
      }
      max = Math.max(max, count);
    }
    return max;
  }

  public static boolean canConstruct(String ransomNote, String magazine) {
    Map<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < magazine.length(); i++) {
      char m = magazine.charAt(i);
      int count = map.getOrDefault(m, 0);
      map.put(m, count + 1);
    }

    for (int i = 0; i < ransomNote.length(); i++) {
      char r = ransomNote.charAt(i);
      int count = map.getOrDefault(r, 0);

      if (count == 0) {
        return false;
      }

      map.put(r, count - 1);
    }
    return true;
  }

  static void toNRecursive(int n) {
    if (n == 0) {
      return;
    }
    toNRecursive(n - 1);
    System.out.println(n);
  }

  public TreeNode sortedArrayToBST(int[] nums) {
    return helper(nums, 0, nums.length - 1);
  }

  public TreeNode helper(int[] nums, int left, int right) {
    if (left > right) {
      return null;
    }
    int mid = (left + right) / 2;
    TreeNode root = new TreeNode(nums[mid]);
    root.left = helper(nums, left, mid - 1);
    root.right = helper(nums, mid + 1, right);
    return root;
  }

  public boolean isSameTree(TreeNode p, TreeNode q) {
    if (p == null && q == null) {
      return true;
    }
    if (p == null || q == null) {
      return false;
    }
    if (p.val != q.val) {
      return false;
    }
    return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
  }

  public int arrayPairSum(int[] nums) {
    Arrays.sort(nums);
    int sum = 0;
    for (int i = 0; i < nums.length; i += 2) {
      sum += nums[i];
    }
    return sum;
  }

  public int reverse(int x) {
    int rev = 0;
    while (x != 0) {
      int pop = x % 10;
      x /= 10;
      if (rev > Integer.MAX_VALUE / 10 || (rev == Integer.MAX_VALUE / 10 && pop > 7))
        return 0;
      if (rev < Integer.MIN_VALUE / 10 || (rev == Integer.MIN_VALUE / 10 && pop < -8))
        return 0;
      rev = rev * 10 + pop;
    }
    return rev;
  }

  public static int numSteps(String s) {
    // doesn't work for big numbers
    int fromStringToInt = Integer.parseInt(s, 2);
    int count = 0;
    while (fromStringToInt != 1) {
      if (fromStringToInt % 2 == 0) {
        fromStringToInt = fromStringToInt / 2;
        count++;
      } else {
        fromStringToInt = fromStringToInt + 1;
        count++;
      }
    }
    return count;
  }

  public boolean checkString(String s) {
    return !s.contains(("ba"));
  }

  public static boolean areNumbersAscending(String s) {
    String[] words = s.split(" ");
    int prev = 0;
    for (String word : words) {
      if (word.matches("\\d+")) {
        int num = Integer.parseInt(word);
        if (num <= prev) {
          return false;
        }
        prev = num;
      }
    }
    return true;
  }

  public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) {
      return result;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while (!queue.isEmpty()) {
      List<Integer> level = new ArrayList<>();
      int size = queue.size();
      for (int i = 0; i < size; i++) {
        TreeNode node = queue.poll();
        if (node != null) {
          level.add(node.val);
        }
        if (node.left != null) {
          queue.add(node.left);
        }
        if (node.right != null) {
          queue.add(node.right);
        }
      }
      result.add(level);
    }
    return result;
  }

  public int minDepth(TreeNode root) {
    int minDepth = 0;
    if (root == null) {
      return minDepth;
    }
    if (root.left == null && root.right == null)
      return 1;
    if (root.left == null)
      return minDepth(root.right) + 1;

    if (root.right == null)
      return minDepth(root.left) + 1;
    return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
  }

  public int maxDepth(TreeNode root) {
    int maxDepth = 0;
    // if (root == null) {
    // return maxDepth;
    // }
    // Queue<TreeNode> queue = new LinkedList<>();
    // queue.add(root);
    // while (!queue.isEmpty()) {
    // int size = queue.size();
    // while (size > 0) {
    // TreeNode node = queue.poll();
    // if (node.left != null) {
    // queue.add(node.left);
    // }
    // if (node.right != null) {
    // queue.add(node.right);
    // }
    // size--;
    // }
    // maxDepth++;
    // }
    if (root == null) {
      return maxDepth;
    }

    return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
  }

  public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    helperInorder(root, res);
    return res;
  }

  public void helperInorder(TreeNode root, List<Integer> res) {
    if (root != null) {
      helperInorder(root.left, res);
      res.add(root.val);
      helperInorder(root.right, res);
    }
  }

  public List<Integer> postOrderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    helperPostOrder(root, res);
    return res;
  }

  public void helperPostOrder(TreeNode root, List<Integer> res) {
    if (root != null) {
      helperPostOrder(root.left, res);
      helperPostOrder(root.right, res);
      res.add(root.val);
    }
  }

  public List<Integer> preOrderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    helperPreOrder(root, res);
    return res;
  }

  public void helperPreOrder(TreeNode root, List<Integer> res) {
    if (root != null) {
      res.add(root.val);
      helperPreOrder(root.left, res);
      helperPreOrder(root.right, res);
    }
  }

  // public static List<List<Integer>> generate(int numRows) {
  // int always = 1;
  // List<List<Integer>> list = new ArrayList<>();
  // for (int i = 0; i < numRows - 1; i++) {
  //
  // }
  // }

  public static int singleNumber(int[] nums) {
    int result = 0;
    for (int i = 0; i < nums.length; i++) {
      result ^= nums[i];
    }
    return result;
  }

  // public static int[] searchRange(int[] nums, int target) {
  // int right = 0;
  // int left = nums.length - 1;
  // int pivot;
  // int[] ans = new int[2];
  //
  // while (right <= left) {
  // pivot = left + (right - left) / 2;
  // if (nums[pivot] == target) {
  // ans[0] = pivot;
  // ans[1] = pivot + 1;
  // return ans;
  // } else {
  // if (target < nums[pivot]) {
  // System.out.println("oof");
  // right = pivot - 1;
  // } else if (target > nums[pivot]) {
  // left = pivot + 1;
  // } else {
  // ans[0] = -1;
  // ans[1] = -1;
  // }
  // }
  // }
  // return ans;
  // }

  public boolean isAnagram(String s, String t) {
    Set<char[]> set = new HashSet<>();
    set.add(s.toCharArray());
    if (!set.add(t.toCharArray())) {
      return false;
    }
    return true;
  }

  public static int[] countBits(int n) {
    int[] ans = new int[n + 1];
    for (int i = 0; i <= n; i++) {
      ans[i] = Integer.bitCount(i);
    }
    return ans;
  }

  public static boolean isPalindrome(String s) {
    if (s.length() == 0 || s.length() == 1)
      return true;
    s = s.trim().toLowerCase().replaceAll("[^a-zA-Z\\d]", "");

    int start = 0;
    int end = s.length() - 1;
    while (start < end) {
      if (s.charAt(start++) != s.charAt(end--)) {
        return false;
      }
    }
    return true;
  }

  // public static int removeDuplicates(int[] nums) {
  // int i = 0;
  // for (int j = 1; j < nums.length; j++) {
  // if (nums[i] != nums[j]) {
  // //i is actually how many were found
  // i++;
  // nums[i] = nums[j];
  // }
  // }
  // return i + 1;
  // }

  public boolean isSymmetric(TreeNode root) {
    return checkForSymmetry(root, root);
  }

  public boolean checkForSymmetry(TreeNode t1, TreeNode t2) {
    if (t1 == null && t2 == null)
      return true;
    if (t1 == null || t2 == null)
      return false;

    return (t1.val == t2.val)
        && checkForSymmetry(t1.left, t2.right)
        && checkForSymmetry(t1.right, t2.left);
  }

  public static int diagonalSum(int[][] mat) {
    // pe randuri
    // for(int i = 0; i < matrix.length; i++) {
    // for(int j = 0; j < matrix[i].length; j++) {
    // System.out.println(matrix[i][j]);
    // }
    // }

    // pe coloane
    // for(int i = 0; i < matrix[0].length; i++) {
    // for(int j = 0; j < matrix.length; j++) {
    // System.out.println(matrix[j][i]);
    // }
    // }

    // for (int i = 0; i < matrix.length; i++) {}
    // {
    // // Printing principal diagonal
    // System.out.print(mat[i][i] + ", ");
    // }

    // int k = n - 1;
    // for (int i = 0; i < matrix.length; i++)
    // {
    // // Printing secondary diagonal
    // System.out.print(mat[i][k--] + ", ");
    // }
    if (mat.length == 1)
      return mat[0][0];
    int sum = 0;
    int k = mat.length - 1;
    for (int i = 0; i < mat.length; i++) {
      sum += mat[i][i] + mat[i][k - i];
    }
    if (mat.length % 2 == 1) {
      sum -= mat[mat.length / 2][mat.length / 2];
    }
    return sum;
  }

  public static int[] countPoints(int[][] points, int[][] queries) {
    // d sqr = (x2 - x1)sqr + (y2 - y1)sqr
    // if whats after = is <= to d sqr, then its inside the circle
    List<Integer> howManyPointsInsideTheCircle = new ArrayList<>();
    int count = 0;

    for (int[] queriesArray : queries) {
      for (int[] pointsArray : points) {
        int radius = ((queriesArray[0] - pointsArray[0]) * (queriesArray[0] - pointsArray[0])
            + (queriesArray[1] - pointsArray[1]) * (queriesArray[1] - pointsArray[1]));
        if (radius <= (queriesArray[2] * queriesArray[2]))
          count++;
      }
      howManyPointsInsideTheCircle.add(count);
      count = 0;
    }
    return howManyPointsInsideTheCircle.stream().mapToInt(Integer::intValue).toArray();
  }

  public static boolean isIsomorphic(String s, String t) {
    HashMap<Character, Character> forS = new HashMap<>();
    HashMap<Character, Character> forT = new HashMap<>();

    for (int i = 0; i < s.length(); i++) {
      char c1 = s.charAt(i);
      char c2 = t.charAt(i);

      if (forS.containsKey(c1) && forS.get(c1) != c2) {
        return false;
      }
      if (forT.containsKey(c2) && forT.get(c2) != c1) {
        return false;
      }
      forS.put(c1, c2);
      forT.put(c2, c1);
    }
    return true;
  }

  public static int climbStairs(int n) {
    int one = 1;
    int two = 1;
    for (int i = 0; i < n - 1; i++) {
      int temp = one;
      one = one + two;
      two = temp;
    }
    return one;
  }

  public static int[] plusOne(int[] digits) {
    for (int i = digits.length - 1; i >= 0; i--) {
      if (digits[i] == 9) {
        digits[i] = 0;
      } else {
        digits[i] += 1;
        return digits;
      }
    }
    int[] arr = new int[digits.length + 1];
    arr[0] = 1;
    return arr;
  }

  public static int pivotIndex(int[] nums) {
    int right = 0;
    for (int num : nums) {
      right += num;
    }
    right -= nums[0];
    int left = 0;
    if (left == right) {
      return 0;
    }
    for (int i = 1; i < nums.length; i++) {
      left += nums[i - 1];
      right -= nums[i];
      if (left == right) {
        return i;
      }
    }
    return -1;
  }

  public static int trailingZeroes(int n) {
    // ologn time
    int result = 0;
    while (n != 0) {
      n = n / 5;
      result += n;
    }
    return result;
  }

  // public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
  // if (headA == null || headB == null) {
  // return null;
  // }
  // while(headA != null || headB != null) {
  // if(headA.val == headB.val) {
  // return headA;
  // }
  // headA = headA.next;
  // headB = headB.next;
  // }
  // return null;
  // }

  public static int countPairs(int[] nums, int k) {
    int count = 0;
    for (int i = 0; i < nums.length; i++) {
      for (int j = i + 1; j < nums.length; j++) {
        if ((nums[i] == nums[j]) && (i * j) % k == 0) {
          count++;
        }
      }
    }
    return count;
  }

  public static int minMovesToSeat(int[] seats, int[] students) {
    // GREEDY APPROACH
    seats = Arrays.stream(seats).sorted().toArray();
    students = Arrays.stream(students).sorted().toArray();
    int moves = 0;
    for (int i = 0; i < seats.length; i++) {
      moves += Math.abs(students[i] - seats[i]);
    }
    return moves;
  }

  public static int countKDifference(int[] nums, int k) {
    int count = 0;
    for (int i = 0; i < nums.length - 1; i++) {
      for (int j = i + 1; j < nums.length; j++) {
        if (Math.abs(nums[i] - nums[j]) == k) {
          count++;
        }
      }
    }
    return count;
  }

  public static int mostWordsFound(String[] sentences) {
    int mostWordsFound = 0;
    for (String sentence : sentences) {
      int count = 0;
      for (String word : sentence.split(" ")) {
        count++;
        mostWordsFound = Math.max(mostWordsFound, count);
      }
    }
    return mostWordsFound;
  }

  public static int finalValueAfterOperations(String[] operations) {
    int result = 0;
    for (String operation : operations) {
      if (operation.equals("--X") || operation.equals("X--")) {
        result--;
      }
      if (operation.equals("++X") || operation.equals("X++")) {
        result++;
      }
    }
    return result;
  }

  public static int[] getConcatenation(int[] nums) {
    int[] arr = new int[nums.length * 2];
    // System.arraycopy(nums, 0, arr, 0, nums.length);
    // System.arraycopy(nums, 0, arr, arr.length/2, nums.length);
    // return arr;
    for (int i = 0; i < nums.length; i++) {
      arr[i] = nums[i];
      arr[i + nums.length] = nums[i];
    }
    return arr;
  }

  public static int[] buildArray(int[] nums) {
    int[] arr = new int[nums.length];
    for (int i = 0; i <= nums.length - 1; i++) {
      arr[i] = nums[nums[i]];
    }
    return arr;
  }

  public int removeElement(int[] nums, int val) {
    int i = 0;
    for (int j = 0; j < nums.length; j++) {
      if (nums[j] != val) {
        nums[i++] = nums[j];
      }
    }
    return i;
  }

  public static int strStr(String haystack, String needle) {
    if (needle.length() == 0) {
      return 0;
    }
    if (haystack.contains(needle)) {
      return haystack.indexOf(needle);
    }
    return -1;
  }

  // public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
  // l1 = reverseLinkedList(l1);
  // l2 = reverseLinkedList(l2);
  // ListNode head = new ListNode(0);
  // ListNode current = head;
  // StringBuilder sb1 = new StringBuilder();
  // StringBuilder sb2 = new StringBuilder();
  // StringBuilder ans;
  // while(l1 != null || l2 != null) {
  // if(l1 != null) {
  // sb1.append(l1.val);
  // l1 = l1.next;
  // }
  // if(l2 != null) {
  // sb2.append(l2.val);
  // l2 = l2.next;
  // }
  // }
  // ans = new StringBuilder(String.valueOf(Integer.parseInt(sb1.toString()) +
  // Integer.parseInt(sb2.toString())));
  // ans.reverse();
  // current.next = new ListNode(Integer.parseInt(String.valueOf(ans)));
  //// ListNode current = head;
  //// int carry = 0;
  //// while(l1 != null || l2 != null) {
  //// int sum = 0;
  //// if(l1 != null) {
  //// sum += l1.val;
  //// l1 = l1.next;
  //// }
  //// if(l2 != null) {
  //// sum += l2.val;
  //// l2 = l2.next;
  //// }
  ////
  //// }
  ////
  // return head.next;
  // }

  // public static int searchx(int[] nums, int target) {
  //
  // https://leetcode.com/problems/search-in-rotated-sorted-array/discuss/1824717/Java-Clear-easy-to-understand-solution
  // }

  public ListNode deleteDuplicates(ListNode head) {
    ListNode current = head;
    while (current != null && current.next != null) {
      if (current.val == current.next.val) {
        current.next = current.next.next;
      } else {
        current = current.next;
      }
    }
    return head;
  }

  public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    ListNode head = new ListNode();
    ListNode current = head;
    while (list1 != null && list2 != null) {
      if (list1.val < list2.val) {
        current.next = list1;
        list1 = list1.next;
      } else {
        current.next = list2;
        list2 = list2.next;
      }
      current = current.next;
    }
    if (list1 != null) {
      current.next = list1;
    }
    if (list2 != null) {
      current.next = list2;
    }
    return head.next;
  }

  public static int squareRootNoInbuiltMethod(double n) {
    // TLE
    // double squareRoot = n/2;
    // double t;
    //
    // do{
    // t = squareRoot;
    // squareRoot = (t + (n/t))/2;
    // } while(t - squareRoot != 0);

    // newton's method
    if (n < 2)
      return (int) n;
    double squareRoot = n;

    while (squareRoot > n / squareRoot) {
      squareRoot = (squareRoot + n / squareRoot) / 2;
    }

    return (int) squareRoot;
  }

  // public static String timeConversion(String s) {
  // for (int i = 0; i < s.length(); i++) {
  // if(s.contains("PM")){
  // s.replace("PM", "");
  // int hour = Integer.parseInt(s.substring(0, 2));
  // if(hour != 12){
  // hour += 12;
  // }
  // }
  // }
  //
  // }

  public static List<Integer> compareTriplets(List<Integer> a, List<Integer> b) {
    List<Integer> result = new ArrayList<>(2);
    result.add(0);
    result.add(0);
    for (int i = 0; i < 3; i++) {
      if (a.get(i) > b.get(i)) {
        result.set(0, result.get(0) + 1);
      } else {
        if (a.get(i) < b.get(i)) {
          result.set(1, result.get(1) + 1);
        }
      }
    }
    return result;
  }

  public static long aVeryBigSum(List<Long> ar) {
    long sum = 0;
    for (int i = 0; i < ar.size(); i++) {
      sum += ar.get(i);
    }
    return sum;
  }

  public static int simpleArraySum(List<Integer> ar) {
    int sum = 0;
    for (int i = 0; i < ar.size(); i++) {
      sum += ar.get(i);
    }
    return sum;
  }

  public static void plusMinus(List<Integer> arr) {
    int positive = 0;
    int negative = 0;
    int zero = 0;
    for (int i = 0; i < arr.size(); i++) {
      if (arr.get(i) > 0) {
        positive++;
      } else if (arr.get(i) < 0) {
        negative++;
      } else if (arr.get(i) == 0) {
        zero++;
      }
    }
    System.out.println(positive / (double) arr.size());
    System.out.println(negative / (double) arr.size());
    System.out.println(zero / (double) arr.size());
  }

  public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
    if (root1 == null)
      return root2;
    if (root2 == null)
      return root1;

    root1.val += root2.val;
    root1.left = mergeTrees(root1.left, root2.left);
    root1.right = mergeTrees(root1.right, root2.right);

    return root1;
  }

  public static boolean isPalindrome(ListNode head) {
    if (head == null || head.next == null)
      return true;
    ListNode fast = head;
    ListNode slow = head;

    while (fast != null && fast.next != null) {
      fast = fast.next.next;
      slow = slow.next;
    }

    // reverse the right half
    slow = reverseLinkedList(slow);
    fast = head;

    while (slow != null) {
      if (fast.val != slow.val)
        return false;
      slow = slow.next;
      fast = fast.next;
    }

    return true;
  }

  public ListNode reverseBetween(ListNode head, int left, int right) {
    // REVERSE LINKED LIST II
    return new ListNode();
  }

  public static ListNode reverseLinkedList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;
    ListNode next = null;
    while (curr != null) {
      next = curr.next;
      curr.next = prev;
      prev = curr;
      curr = next;
    }
    head = prev;
    return head;
  }

  public static boolean checkInclusion(String s1, String s2) {
    char[] chars1 = s1.toCharArray();
    char[] chars2 = s2.toCharArray();
    Arrays.sort(chars1);
    Arrays.sort(chars2);
    System.out.println(Arrays.toString(chars1));
    System.out.println(Arrays.toString(chars2));
    return Arrays.toString(chars2).contains(Arrays.toString(chars1));
  }

  public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode fake_head = new ListNode(0);
    fake_head.next = head;

    ListNode fast = fake_head;
    ListNode slow = fake_head;

    for (int i = 1; i <= n + 1; i++) {
      fast = fast.next;
    }
    while (fast != null) {
      slow = slow.next;
      fast = fast.next;
    }
    slow.next = slow.next.next;
    return fake_head.next;
  }

  public static ListNode removeElements(ListNode head, int val) {
    // pt cand nr cautat e chiar head
    while (head != null && head.val == val) {
      head = head.next;
    }

    ListNode iter = head;
    while (iter != null && iter.next != null) {
      if (iter.next.val == val) {
        iter.next = iter.next.next;
      } else {
        iter = iter.next;
      }
    }

    return head;
  }

  public void deleteNode(ListNode node) {
    node.val = node.next.val;
    node.next = node.next.next;
  }

  public static ListNode middleNode(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;
    while (fast != null && fast.next != null) {
      slow = slow.next;
      fast = fast.next.next;
    }
    return slow;
  }

  public ListNode deleteMiddle(ListNode head) {
    if (head.next == null)
      return null;

    // edge case for 2 nodes
    if (head.next.next == null) {
      head.next = null;
      return head;
    }

    ListNode slow = head;
    ListNode fast = head;
    while (fast != null && fast.next != null) {
      slow = slow.next;
      fast = fast.next.next;
    }
    assert slow != null;
    assert slow.next != null;
    slow.val = slow.next.val;
    slow.next = slow.next.next;
    return head;
  }

  public static String reverseWordsJUSTHECHARS(String s) {
    // StringBuilder sb = new StringBuilder();
    // String[] words = s.split(" ");
    // for(String word : words) {
    // sb.append(new StringBuilder(word).reverse());
    // sb.append(" ");
    // }
    // return sb.toString().stripTrailing();
    String[] words = s.split(" ");
    StringBuilder sb = new StringBuilder();
    for (String word : words) {
      char[] character = word.toCharArray();
      int start = 0;
      int end = character.length - 1;
      for (int i = start; i < end; i++) {
        char temp = character[start];
        character[start] = character[end];
        character[end] = temp;

        start++;
        end--;
      }
      sb.append(" ").append(character);
    }

    return sb.toString().trim();
  }

  public static void reverseStringInPlace(char[] s) {
    // int start = 0;
    // int end = s.length-1;
    // for(int i = start; i < end; i++){
    // char temp = s[start];
    // s[start] = s[end];
    // s[end] = temp;
    //
    // start++;
    // end--;
    // }
    StringBuilder sb = new StringBuilder();
    sb.append(s);
    sb.reverse();
    for (int i = 0; i < s.length; i++) {
      s[i] = sb.charAt(i);
    }
    System.out.println(s);
  }

  public static int prodImpare() {
    Scanner scanner = new Scanner(System.in);
    int n = scanner.nextInt();
    int produs = 1;
    for (int i = 1; i <= 2 * n - 1; i += 2) {
      produs = produs * i;
    }
    return produs;
  }

  public static int[] intersect(int[] nums1, int[] nums2) {
    Arrays.sort(nums1);
    Arrays.sort(nums2);
    int top = 0;
    int bottom = 0;
    List<Integer> result = new ArrayList<>();
    while (true) {
      if (top >= nums1.length || bottom >= nums2.length) {
        break;
      }
      if (nums1[top] == nums2[bottom]) {
        result.add(nums1[top]);
        top++;
        bottom++;
      } else if (nums1[top] < nums2[bottom]) {
        top++;
      } else if (nums1[top] > nums2[bottom]) {
        bottom++;
      }
    }
    int[] ans = new int[result.size()];
    for (int i = 0; i < result.size(); i++) {
      ans[i] = result.get(i);
    }
    return ans;
  }

  public static void merge(int[] nums1, int m, int[] nums2, int n) {
    // nums1 = IntStream.concat(Arrays.stream(nums1),
    // Arrays.stream(nums2)).filter(number ->
    // number != 0).sorted().toArray();
    // System.out.println(Arrays.toString(nums1));
    for (int i = 0; i < n; i++) {
      nums1[i + m] = nums2[i];
    }
    Arrays.sort(nums1);
    System.out.println(Arrays.toString(nums1));
  }

  public static int[] twoSum2(int[] numbers, int target) {
    boolean found = false;
    int[] result = new int[2];
    int i = 0;
    while (i < numbers.length - 1 && !found) {
      int j = i + 1;
      while (j < numbers.length && !found) {
        if (numbers[i] + numbers[j] == target) {
          result[0] = i + 1;
          result[1] = j + 1;
          found = true;
        }
        j++;
      }
      i++;
    }
    return result;
  }

  public static void moveZeroes(int[] nums) {
    int index = 0;
    for (int i = 0; i < nums.length; i++) {
      if (nums[i] != 0) {
        nums[index++] = nums[i];
      }
    }
    for (int i = index; i < nums.length; i++) {
      nums[i] = 0;
    }
  }

  public static int maxSubArray(int[] nums) {
    int maxSub = nums[0];
    int curSum = 0;
    for (int i = 0; i < nums.length; i++) {
      if (curSum < 0) {
        curSum = 0;
      }
      curSum += nums[i];
      maxSub = Math.max(maxSub, curSum);
    }
    return maxSub;
  }

  public static void rotate(int[] nums, int k) {
    // It might help to think of the array as a circle. Let’s say the array has
    // length 5, and
    // contains the elements: A = 1, 2, 3, 4, 5.
    // If k = 1, then rotate our circle by 1 position, i.e., A’ = 5, 1, 2, 3, 4.
    // If k = 5 what happens? We rotate our circle 5 times, and we are back where we
    // started.
    k = k % nums.length;
    // reverse all numbers 7 6 5 4 3 2 1
    reverse(nums, 0, nums.length - 1);
    // reverse first k numbers 5 6 7 4 3 2 1
    reverse(nums, 0, k - 1);
    // reverse last k numbers 5 6 7 1 2 3 4
    reverse(nums, k, nums.length - 1);
    System.out.println(Arrays.toString(nums));
  }

  private static void reverse(int[] nums, int start, int end) {
    while (start < end) {
      int temp = nums[start];
      nums[start] = nums[end];
      nums[end] = temp;
      start++;
      end--;
    }
  }

  public static int[] sortedSquares(int[] nums) {
    for (int i = 0; i < nums.length; i++) {
      nums[i] = nums[i] * nums[i];
    }
    Arrays.sort(nums);
    return nums;
  }

  public static int searchInsert(int[] nums, int target) {
    int left = 0;
    int right = nums.length - 1;
    int pivot;
    int whereItWouldBe = 0;
    while (left <= right) {
      pivot = left + (right - left) / 2;
      if (nums[pivot] == target) {
        return pivot;
      } else {
        if (target < nums[pivot]) {
          whereItWouldBe = pivot;
          right = pivot - 1;
        } else {
          whereItWouldBe = pivot + 1;
          left = pivot + 1;
        }
      }
    }
    return whereItWouldBe;
  }

  public static int search(int[] nums, int target) {
    // BASIC BINARY SEARCH 1 2 3 4 5
    int left = 0;
    int right = nums.length - 1;
    int pivot;
    while (left <= right) {
      pivot = (left + right) / 2; // the middle
      // better use pivot = left + (right - left) /2
      if (nums[pivot] == target) {
        return pivot;
      } else {
        if (target < nums[pivot]) {
          right = pivot - 1;
        } else {
          left = pivot + 1;
        }
      }
    }
    return -1;
  }

  public static int[] productExceptSelf(int[] nums) {
    // w DIVISION
    // int[] answer = new int[nums.length];
    // int one = 1;
    // for(int i = 0; i < nums.length; i++) {
    // one = one * nums[i];
    // }
    // for(int i = 0; i < nums.length; i++){
    // nums[i] = one / nums[i];
    // answer[i] = nums[i];
    // }
    // return answer;

    // w/o DIVISION, cu left si right
    // partea din stanga lui i, de la 1 la sfarsit
    int[] leftSide = new int[nums.length];
    leftSide[0] = 1;
    for (int i = 1; i < nums.length; i++) {
      leftSide[i] = leftSide[i - 1] * nums[i - 1];
    }

    // partea din dreapta lui i, de la coada la capat
    int[] rightSide = new int[nums.length];
    rightSide[nums.length - 1] = 1;
    for (int i = nums.length - 2; i >= 0; i--) {
      rightSide[i] = rightSide[i + 1] * nums[i + 1];
    }

    // inmultesti left cu right
    for (int i = 0; i < nums.length; i++) {
      nums[i] = leftSide[i] * rightSide[i];
    }
    return nums;
  }

  public static boolean containsDuplicate(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int i : nums) {
      if (!set.add(i)) {
        return true;
      }
    }
    return false;
  }

  public static int maxProfit(int[] prices) {
    if (prices.length == 0 || prices == null) {
      return 0;
    }
    int minPrice = prices[0];
    int profit = 0;
    for (int price : prices) {
      if (price < minPrice) {
        minPrice = price;
      } else {
        profit = Math.max(profit, price - minPrice);
      }
    }
    return profit;
  }

  public static String longestPalindrome(String s) {
    if (s == null || s.length() <= 1) {
      return s;
    }

    String longest = s.substring(0, 1);
    for (int i = 0; i < s.length(); i++) {
      String temp = palindromeHelper(s, i, i); // odd length ex: racecar
      if (temp.length() > longest.length()) {
        longest = temp;
      }

      temp = palindromeHelper(s, i, i + 1); // even length ex: abba
      if (temp.length() > longest.length()) {
        longest = temp;
      }
    }
    return longest;
  }

  // given a string and the CENTER(S) of the palindrome (i and i+1), find the
  // longest possible
  // palindrome
  // from center, go left and right till characters are same and return the
  // longest palindromic
  // substring
  private static String palindromeHelper(String string, int begin, int end) {
    while (begin >= 0 && end < string.length() && string.charAt(begin) == string.charAt(end)) {
      begin--;
      end++;
    }
    return string.substring(begin + 1, end);
  }

  public static int lengthOfLongestSubstring(String s) {
    Set<Character> chars = new HashSet<>();
    int i = 0;
    int j = 0;
    int maxLength = 0;
    while (j < s.length()) {
      if (chars.contains(s.charAt(j))) {
        chars.remove(s.charAt(i++));
      } else {
        chars.add(s.charAt(j++));
        maxLength = Math.max(maxLength, chars.size());
      }
    }
    return maxLength;
  }

  public static int[] twoSum(int[] nums, int target) {
    int[] ans = new int[2];
    boolean found = false;
    int i = 0;
    while (i < nums.length - 1 && !found) {
      int j = i + 1;
      while (j < nums.length && !found) {
        if (nums[i] + nums[j] == target) {
          ans[0] = i;
          ans[1] = j;
          found = true;
        }
        j++;
      }
      i++;
    }
    return ans;
  }
}
