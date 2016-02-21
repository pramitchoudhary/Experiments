import java.util.*;

class FindPath {

    public enum State {
	unvisited, visited, visiting;}
    
    public static boolean search(Graph g, Node start, Node end) {
	Queue<Node> q = new LinkedList<Node>();

	for(Node u : g.getNodes()) {
	    u.state = State.unvisited;
	}

	start.state = State.visiting;
	// add to queue
	q.add(start);
	
	while(!q.isEmpty()){
	    Node u = q.pop();
	    if(u != null) {
		for(Node v: u.getAdjacent()){
		    if(v == end) {
			return true;
		    }
		    else {
			v.state = State.visiting;
			q.add(v);
		    }
		}
	    }
	    u.state = State.visited;
	}
	return false;
    }
}
