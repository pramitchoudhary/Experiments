import java.util.*;
import java.lang.Math;

class ITOA {
    public static String convert(int input, int base)
    {
	if(input == 0) return String.valueOf(input);
	String s = "";
	int x = Math.abs(input);
	while(x > 0 ) {
	    s = (x%base) + s;
	    x = x/base;
	}

	if(input < 0) return "-" + s;
	else return s;	   
    }

    public static void main(String[] args)
    {
	System.out.println(convert(7, 2));
    }
    
}
