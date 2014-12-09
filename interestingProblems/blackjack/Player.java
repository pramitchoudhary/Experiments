/*
 * Simulate a black-jack Player
 */

import java.util.*;
import java.math.*;
import com.google.common.math.*;

public class Player
{
        /*
         * Name of the player
         */
        private String name;
        
        /*
         * Number of cards in a player's current hand
         */
        private ArrayList<Card> hand;

        private Deck dInstance;

        //private ArrayList<Pair<int,int>> ;
        /**
         * @param  pName Name of the player
         */
        public Player(String pName, Deck dObj){

                this.name = pName;
                hand = new ArrayList<Card>();
                dInstance = dObj;
                //combinations = ArrayList<Pair<int,int>>();
        }

        /**
         * Reset player's hand to zero
         */
        public void resetHand(){
                hand.clear();
        }

        /**
         * Count the Prob of being Under or Over in a Game of Black Jack
         */
        public void countCards()
        {

                HashMap<Integer, Integer> currentHandDict = dInstance.getCurrentDeckMap();
                float underOrEqual = 0;
                float over = 0;

                int currentSum = computeHandSum();
                int diff = 21 - currentSum;
                int diff2=0;
                if ( diff > 10 )
                        {
                                diff = 10;
                        }
                int possibleCardChoiceCount = 0;
                for (int k=1; k<=diff; k++)
                        {
                                possibleCardChoiceCount+= currentHandDict.get(k);
                        }
                
                underOrEqual = (float) possibleCardChoiceCount/dInstance.getDeckCount();
                over = 1 - underOrEqual;

                System.out.println("Probability of being Under: " + underOrEqual);
                System.out.println("Probability of being Over: " + over);
        }
        
       /**
         * Computes Probability to get Black Jack
         * @return float Odds of getting a black Jack on each try
         */
        public float computeProbToGetBlackJack(int numOfDeck)
        {

                HashMap<Integer, Integer> currentHandDict = dInstance.getCurrentDeckMap();

                // Default Probability of getting a black-jack at the start of the game
                int cardsOnDeck = numOfDeck*52;
                int numAcesRemaining = numOfDeck*4;
                int tenValuedCardRemaining = numOfDeck*16; // < 10s:4; K:4; Q:4; J:4 >
                float combinationCount = BigIntegerMath.binomial(cardsOnDeck, 2).floatValue();
                float defaultProb  = (float) (numAcesRemaining * tenValuedCardRemaining) /combinationCount;

                // Odds calculated after each attempt
                int currentSum = computeHandSum();
                int sumTo21 = 21 - currentSum;
                int[] defaultSet = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
                float oddsValue =0;
                cardsOnDeck = dInstance.getDeckCount();
                if( sumTo21 >= 10 )
                        {
                                tenValuedCardRemaining = currentHandDict.get(10);
                                numAcesRemaining = currentHandDict.get(1);
                                oddsValue = (float) (numAcesRemaining + tenValuedCardRemaining)/cardsOnDeck;
                        }
                else if (  sumTo21 >0 && sumTo21 < 10)
                            {
                                   float faceCardRemaining = currentHandDict.get(sumTo21);
                                    oddsValue = (float) faceCardRemaining/cardsOnDeck;
                            }

                System.out.println("Probability of getting BlackJack at the start of the game: " + defaultProb);
                return oddsValue;
        }
        
        /**
         * Reset player's hand to zero
         * @return int Sum of the cards at the end of a hand
         */
        public int computeHandSum(){
                int sum = 0;
                int nAces = 0;

                for( Card c: hand ){
                        if( c.isAce() ) nAces++;
                        int cardValue = c.getNumber();
                        sum+= cardValue;
                }

                while (sum > 21 && nAces >0 )
                        {
                                sum = sum +10; // +10 bcauz -1, +10 for each ace
                                nAces--;
                       }
                
                return sum;
        }
        
        /**
         * Add a card to Player's hand
         * @param newCard the card object to be added
         * @return whether the sum of the current hand is < or > 21
         */
        public boolean addCard( Card newCard){
                if ( hand.size() == 10 )
                        System.out.println(" Player already has 10 cards, cannot add more");
                else
                        this.hand.add(newCard);

                return ( computeHandSum()  >= 21 );
        }

        /**
         * Print the player's card
         * @param displayFirstCard flag to keep track if the first card is hidden or not, useful incase the player is the Dealer
         */
        public void printHand( boolean displayFirstCard ){
                System.out.println("Player: " + this.name );

                if (displayFirstCard )
                        System.out.println(hand.get(0).getCardStr() + " of " + hand.get(0).getSuitType().toString());
                else
                        {
                                System.out.println("First card is hidden");
                        }
                for ( int index = 1; index < hand.size(); index++ )
                        {
                                System.out.println(hand.get(index).getCardStr() + " of " + hand.get(index).getSuitType().toString());
                       }
                System.out.println();
        }               
}