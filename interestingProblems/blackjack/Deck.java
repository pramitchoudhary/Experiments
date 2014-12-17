/**
 * An Implementation to track the state of the Deck
 */
import java.util.*;

public class Deck
{
        private ArrayList<Card> cards;
        private int dealtIndex = 0; // undealt card, cards currently in the deck

        private HashMap<Integer, Integer> handDict;
        private int nDecks;

        /**
         * @param  numberOfDecks Number of Decks to Use for the Game
         */
        public void setDeckOfCards(int numberOfDecks)
        {
                cards = new ArrayList<Card>();
                for ( int s=0; s<numberOfDecks*4; s++)
                        {
                                for ( int d =1; d<14; d++){
                                        Card tempObj;
                                        tempObj = new Card(Suit.values()[s %4],d);
                                        cards.add(tempObj);
                                }
                        }
                nDecks = numberOfDecks;
                handDict = new HashMap<Integer, Integer>();
                initializeDeck();
        }

        private void initializeDeck()
        {
                int initialCount = nDecks*4;
                handDict.put(1, initialCount);
                handDict.put(2, initialCount);
                handDict.put(3, initialCount);
                handDict.put(4, initialCount);
                handDict.put(5, initialCount);
                handDict.put(6, initialCount);
                handDict.put(7, initialCount);
                handDict.put(8, initialCount);
                handDict.put(9, initialCount);
                handDict.put(10, initialCount*4);
        }

        public HashMap<Integer, Integer> getCurrentDeckMap()
        {
                return handDict;
        }
        
        /**
         * Shuffle the cards by randomly swapping pairs of cards
         */
        public void shuffle(){
                Random rGen = new Random();

                Card temp;
                int j;
                for (int index = 0; index < this.cards.size(); index++){
                        j = rGen.nextInt(this.cards.size());
                        
                        temp = cards.get(index);
                        this.cards.set(index, this.cards.get(j));
                        this.cards.set( j, temp );
                }
        }

        /**
         * @return the top card as the deal card
         */
        public Card dealNextCard(boolean hidden){

                Card top = cards.get(0);
                cards.remove(0);

                // Update the handDict
                int value =0;
                if (top.isAce()) {
                        value = 1;
                }
                else
                        {
                                value = top.getNumber();
                        }
                int currentCount = 0;
                if (hidden == true)
                        {
                                currentCount = handDict.get( 10 );
                                // Update the deck
                                handDict.put(10, currentCount -1 );
                        }
                else
                        {
                                currentCount = handDict.get( value );
                                // Update the deck
                                handDict.put(top.getNumber(), currentCount -1 );
                        }
                
                return top;
        }

         public void setRemovedCard(Card cardRemoved){
                 cards.add(cardRemoved);
                 int val = cardRemoved.getNumber();
                 handDict.put(val, handDict.get(val) +1 );
         }
        
        /**
         * @param  rangeToDisplay number of cards from the top of the deck to display
         */
        public void diplayCardsOnDeck(int rangeToDisplay){

                int count = 0;
                for (Card obj : cards)
                        {
                                if( count < rangeToDisplay ){
                                        String numStr = obj.getCardStr();
                                        System.out.println( (count +1) + " " + numStr + " of " + obj.getSuitType().toString());
                                        count ++;
                                }
                       }
                System.out.printf("\t\t[%d other]\n", this.cards.size() - (count));
        }

        /**
         * @return Current count of the Deck
         */
        public int getDeckCount()
        {
                return cards.size();
        }
}
