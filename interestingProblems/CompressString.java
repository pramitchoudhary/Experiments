import java.util.*;

class CompressString {
    public static String compression(String input){
	char[] strToCharArray = input.toCharArray();
	HashMap<Character, Integer> h = new LinkedHashMap<Character, Integer>();
	for(int i=0; i< input.length(); i++){
	    
	    if(!h.containsKey(input.charAt(i))) 
	       h.put(input.charAt(i), 1);
	    else {
		h.put(input.charAt(i), h.get(input.charAt(i)) + 1);
	    }
	}

	// Iterate over the LinkedHasMap and build a String
	StringBuilder sb = new StringBuilder();
	Iterator it = h.entrySet().iterator();
	while(it.hasNext()){
	    Map.Entry pair = (Map.Entry)it.next();
	    sb.append(pair.getKey());
	    sb.append(pair.getValue());
	}
	return sb.toString();
    }
    
    public static void main(String[] args){
	System.out.println(compression("aabcccccaa"));
    }
}
