# Black-Jack
This is a basic implementation of the famous Black-Jack. Currently,  it's only CLI based.
### Features Implemented:
    * Size of the deck be specified at run-time.
    * Random shuffing of the deck is done before the game starts
    * Initially the game starts with initializing 2 hands both for the Dealer and the Player
    * Dealer's first hand is kept hidden, this is changed to display the cards owned by the Dealer. Otherwise, dealer reveals his card only at the end.
    * User has the option to keep it's card face-up or face-down    
    * The decision to "Hit"/"Stay" is implicitly suggested based on the Probability value
### Probability of wining is calculated at each hand for the Player:
    * Stats displayed to the user:
    1. Under and Over 21: This displays the probability of staying under or going over after each hand
    2. Probability of getting 21 or max value close to 21 at the end of each try
    3. It also displays the default chance of getting perfect 21 with the combination of (faceCard + Ace) at the start of the game

### Usage:
##### Build:
    1. Fork or download the repository in a local directory
    2. Download the guava-18.0.jar from http://mvnrepository.com/artifact/com.google.guava/guava/18.0
    3. If you wish to build the source from scratch, execute the following command at CLI
       
       * javac -g Suit.java Card.java Deck.java
       * javac -g -cp guava-18.0.jar:. Player.java GameManager.java
       * Execute: java -cp guava-18.0.jar:. GameManager
       
#####Without Building:
    Download 'blackjack.jar and guava-18.0.jar' and execute 'java -jar  blackjack.jar'

### Future Implementation
1. Add support for Web
2. Compute probability based on pre-computed matrix as suggested at the following link:
http://en.wikipedia.org/wiki/Blackjack#Basic_strategy
3. Add support for splitting, Double and others




