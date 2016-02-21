import java.util.*;

class TreeFunctions {
    public int htOfTree(TreeNode root) {
	if(root == null) return 0;
	return Math.max(htOfTree(root.left), htOfTree(root.right)) + 1;
    }

    public int htOfTreeIteratively(TreeNode root) {
	Queue<TreeNode> q = new LinkedList<TreeNode>();
	q.add(root);
	int ht = 1;
	while(!q.isEmpty()){
	    h+=1;
	    size = q.size()
	    while(size > 0 ){
		TreeNode currentNode = q.pop();
		if(currentNode.left!=null) q.add(currentNode.left);
		if(currentNode.right!=null) q.add(currentNode.right);
		size--;
	    }
	}
    }
    
    public static void main(String[] args)
    {
       
    }
}
