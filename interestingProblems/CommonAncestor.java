import java.util.*;

class CommonAncestor {

    public static boolean helper(TreeNode root, TreeNode childNode) {
	if(root == null) return false;
	if(root == childNode) return true;
	return helper(root.left, childNode) || helper(root.right, childNode);
    }
    
    public static TreeNode commonNode(TreeNode root, TreeNode p, TreeNode q){
	if(root == null) return null;
	if(p == null || q == null) return root;

	// if p is on the left or right of the root
	boolean isPOnLeft = helper(root.left, p); // if this returns false, p is on the right
	boolean isQOnRight = helper(root.right, q); // if this returns false q is on the left

	// p and q on left
	if(isPOnLeft && !isQOnRight){
	    commonNode(root.left, p, q);
	}// p on right and q on right
	else if(!isPOnLeft && isQOnRight) {
	    commonNode(root.right, p, q);
	} // both on opposite sides
	else {
	    return root;
	}
			     
    }
    
}
