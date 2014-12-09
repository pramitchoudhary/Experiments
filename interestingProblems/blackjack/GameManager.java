/*
 * Class to Manage and Monitor the Game
 */

import java.util.Scanner;
import java.io.IOException;

public class GameManager{
        private Deck deckObj;
        public Scanner scObj; 

        public GameManager(){
                deckObj = new Deck();
                scObj = new Scanner(System.in);
        }

        public Deck getDeckInstance()
        {
                return deckObj;
        }
        
        public static void main(String[] args)
        {
               
                GameManager gm = new GameManager();
                Deck deckInstance = gm.getDeckInstance();

                String playerName = null;
                System.out.println("Enter the name of the player");
                try{
                       playerName = gm.scObj.next();
                }
                catch(Exception e)
                        {
                                System.err.println(e.toString());
                                e.printStackTrace();
                        }
                System.out.println();
                
                Player me = new Player(playerName, deckInstance);
                Player dealer = new Player("Dealer", deckInstance);
                boolean playerDone = false;
                boolean dealerDone = false;
                String inputChoice;

                 System.out.println("Enter the number of decks to use");
                 int numOfDeck = Integer.parseInt( gm.scObj.next() );
                 System.out.println();

                 System.out.println("Do you want your cards to be hidden till you hit(true/false)");
                 boolean hide = Boolean.valueOf( gm.scObj.next() );
                 System.out.println();
                  
                //1. Set the deck of cards
                deckInstance.setDeckOfCards(numOfDeck);
                
                //Uncomment the below line to take a peak at the Deck, more for debugging purpose
                // deckObj.diplayCardsOnDeck(10);
                
                //2. Shuffle the cards
                deckInstance.shuffle();

                // The game starts by initializing 2 hands
                me.addCard(deckInstance.dealNextCard());
                dealer.addCard(deckInstance.dealNextCard());

                me.addCard(deckInstance.dealNextCard());
                dealer.addCard(deckInstance.dealNextCard());

                // Print initial Hands
                System.out.println("Cards Dealt");
                
                me.printHand(hide);
                dealer.printHand(true);

                while (!playerDone || !dealerDone )
                        {
                                if( !playerDone )
                                        {

                                                System.out.println("Stats of the Game");
                                                me.countCards();
                                                System.out.println("Odds of getting BlackJack: " +  me.computeProbToGetBlackJack(numOfDeck));
                                                
                                                System.out.println("Enter Hit or Stay: ");
                                                inputChoice = gm.scObj.next();
                                                System.out.println();
                                                
                                                if( inputChoice.compareToIgnoreCase("Hit") == 0 ) {
                                                        playerDone = me.addCard(deckInstance.dealNextCard());
                                                        me.printHand(hide);
                                                }
                                                else {
                                                        playerDone = true;
                                                }
                                        }
                                if ( !dealerDone ){ // dealer's turn
                                        if (dealer.computeHandSum() <17){
                                                System.out.println("The dealer hits");
                                                dealerDone = dealer.addCard(deckInstance.dealNextCard());

                                                // Dealer shows the card after each hand
                                                dealer.printHand(true);
                                        }
                                        else{
                                                System.out.println("The Dealer stays ");
                                                dealerDone = true;
                                        }
                                }
                        }

                // Close the Scanner
                gm.scObj.close();

                // Display the final hands
                me.printHand(false);
                dealer.printHand(false);

                // Compute Result
                int playerSum = me.computeHandSum();
                int dealerSum = dealer.computeHandSum();

                if( playerSum  > dealerSum && playerSum <=21 || dealerSum > 21 )
                        {
                                System.out.println("You Win");
                        }
                else
                        {
                                System.out.println("Dealer Wins");
                        }
       }
}
