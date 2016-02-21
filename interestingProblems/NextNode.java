import java.util.*;

class NextNode {

    public static TreeNode leftMostChild(TreeNode root) {
	if(root == null) return null;
	return leftMostChild(root.left);
    }
    
    public static TreeNode next(TreeNode n) {
	// does n have a right sub tree, if yes then left return the left most node
	// if a right subtree is not there, then go up to the parent
	if(n == null) return null;
	if(n.parent == null) return null;
	else if(n.right!=null) return leftMostChild(n.right);
	else {
	    TreeNode q = n;
	    TreeNode x = q.parent;
	    while(x.left! == q) {
		q = x;
		x = x.parent;
	    }
	}
	return x;	   
    }
}
