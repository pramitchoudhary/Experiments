/* Author:Pramit Choudhary */

import java.util.*;

class Ngrams
{
    public ArrayList<String> list = new ArrayList<String>();
    public ArrayList<String> generateGrams(String str, int gramIndex)
    {
    	String[] tokens = str.split(" ");
    	for(int k=0; k<tokens.length - gramIndex + 1; k++)
    	{
    		StringBuilder concat = new StringBuilder();
    		for(int j=0; j<gramIndex; j++)
    		{
    			if((k+j) < tokens.length)
    			{	
    				concat.append(tokens[k+j] + " ");
    			}
    		}
    		list.add(concat.toString());

    	}

    	return list;
    }
    
    public static void main(String[] args)
    {
		String inputStr = "My name is Pramit";
		Ngrams obj = new Ngrams();
		for(int i = 1; i<=4; i++)
		    {
			for(String s: obj.generateGrams(inputStr, i))
			    {
				System.out.println(s);
			    }
			obj.list.clear();	
		    }
    }
}