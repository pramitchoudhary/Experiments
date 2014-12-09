This is a basic implementation of the famous Black-Jack. Currently,  it's only CLI based.
*Features Implemented:
    * Size of the deck be specified at run-time.
    * Random shuffing of the deck is done before the game starts
    * Initially the game starts with initializing 2 hands both for the Dealer and the Player
    * Dealer's first hand is kept hidden, this is changed to display the cards owned by the Dealer. Otherwise, dealer reveals his card only at the end.
    * User has the option to keep it's card face-up or face-down    
    * The decision to "Hit"/"Stay" is implicitly suggested based on the Probability value
* Probability of wining is calculated at each hand for the Player:
    * Stats displayed to the user:
      1. Under and Over 21: This displays the probability of staying under or going over after each hand
      2. Probability of getting 21 or max value close to 21 at the end of each try
      3. It also displays the default chance of getting perfect 21 with the combination of (faceCard + Ace) at the start of the game

