/* An implementation of the card type
 */

import java.util.*;

public class Card
{
        /**
         * Suit as in the Card type
         */
        private Suit cardSuit;

        /**
         *  Face value of the card ( Ace:1, Jack-King ( 11 - 13 )
         */
        private int cardValue;

        private HashMap<Integer,String> dict;
        
        /**
         * @param  type Description. 
         * @param  faceValue Description. 
         */
        public Card(Suit type, int faceValue)
        {
                this.cardSuit = type;
                if( faceValue >=1 && faceValue <=13)
                        this.cardValue = faceValue;
                else
                        {
                                System.err.println("Not a valid card number");
                                System.exit(1);
                        }
                dict = new HashMap<Integer,String>();
                valueToStringMap();
                
        }

        /**
         * @return true if the cardValue is equal to 1
         */
        public boolean isAce()
        {
                return  cardValue == 1;
        }

        public boolean isFaceCard()
        {
                return cardValue >=11 && cardValue <=13;
        }
        
        /**
         * @return HashMap maps Integer value to String representation of the Card Value
         */
        private void valueToStringMap()
        {
                dict.put(1,"Ace");
                dict.put(2,"Two");
                dict.put(3,"Three");
                dict.put(4,"Four");
                dict.put(5,"Five");
                dict.put(6,"Six");
                dict.put(7,"Seven");
                dict.put(8,"Eight");
                dict.put(9,"Nine");
                dict.put(10,"Ten");
                dict.put(11,"Jack");
                dict.put(12,"Queen");
                dict.put(13,"King");
       }
        
        /**
         * @return cardValue the face value of the card
         */
        public int getNumber(){
                if (isAce()) return 11;
                else if ( cardValue >=11 && cardValue <=13 ) return 10;
                else return cardValue;
        }

        public Suit getSuitType()
        {
                return cardSuit;
        }

        public String getCardStr()
        {
                return dict.get( cardValue );
        }

        public int getCardValue()
        {
                return cardValue;
        }
}


