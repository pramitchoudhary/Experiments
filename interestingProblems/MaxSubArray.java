import java.util.*;
import java.lang.*;

class MaxSubArray {
    public static int maxSum(int[] a) {
	int maxEnding = 0;
	int maxSoFar = 0;
	for(int i: a) {
	    maxEnding = Math.max(0, maxEnding + i);
	    maxSoFar = Math.max(maxEnding, maxSoFar);
	}
	return maxSoFar;
    }
	
    public static void main(String[] args) {
	int[] input = {-2,1, -3, 4, -1, 2, 1, -5, 4};
	System.out.println(maxSum(input));
    }
}
