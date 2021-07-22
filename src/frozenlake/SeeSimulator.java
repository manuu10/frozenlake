package frozenlake;

import frozenlake.pfadfinder.IPfadfinder;

public class SeeSimulator {
	public static final String ANSI_RESET = "\u001B[0m";
	public static final String ANSI_BLACK = "\u001B[30m";
	public static final String ANSI_RED = "\u001B[31m";
	public static final String ANSI_GREEN = "\u001B[32m";
	public static final String ANSI_YELLOW = "\u001B[33m";
	public static final String ANSI_BLUE = "\u001B[34m";
	public static final String ANSI_PURPLE = "\u001B[35m";
	public static final String ANSI_CYAN = "\u001B[36m";
	public static final String ANSI_WHITE = "\u001B[37m";

	public static final String ANSI_BLACK_BACKGROUND = "\u001B[40m";
	public static final String ANSI_RED_BACKGROUND = "\u001B[41m";
	public static final String ANSI_GREEN_BACKGROUND = "\u001B[42m";
	public static final String ANSI_YELLOW_BACKGROUND = "\u001B[43m";
	public static final String ANSI_BLUE_BACKGROUND = "\u001B[44m";
	public static final String ANSI_PURPLE_BACKGROUND = "\u001B[45m";
	public static final String ANSI_CYAN_BACKGROUND = "\u001B[46m";
	public static final String ANSI_WHITE_BACKGROUND = "\u001B[47m";

	public static void main(String[] args) {
		int anzahlSchritte = 0;
		boolean useStateValue = false;
		boolean useNN = false;
		boolean useOnPolicy = true;
		int lakeSize = 6;
		boolean useRandomGenerated = true;

		try {
			IPfadfinder joe = new frozenlake.pfadfinder.mustergruppe.Pfadfinder();
			See testSeeRandom = new See("Testsee", lakeSize, new Koordinate(0, 0),
					new Koordinate(lakeSize - 1, lakeSize - 1));
			See testsee = See.ladeSee("D:\\Development\\java\\frozenlake\\testseen\\", "See8");
			if (useRandomGenerated)
				testsee = testSeeRandom;
			testsee.wegErzeugen();
			// testsee.speichereSee("Testsee");
			System.out.println("start learning");
			// Trainieren mit statevalue, mit NN, onpolicy
			joe.lerneSee(testsee, useStateValue, useNN, useOnPolicy);
			System.out.println("finished learning");
			// Testdurchlauf mit dem trainierten IPfadfinder
			joe.starteUeberquerung(testsee, useStateValue, useNN, useOnPolicy);
			testsee.anzeigen();

			Zustand naechsterZustand = Zustand.Start;
			do {
				Richtung r = joe.naechsterSchritt(naechsterZustand);
				System.out.println("Gehe " + r);
				naechsterZustand = testsee.geheNach(r);
				anzahlSchritte++;
				printSeeColored(testsee);
			} while (!((naechsterZustand == Zustand.Ziel) || (naechsterZustand == Zustand.Wasser)
					|| anzahlSchritte > lakeSize * lakeSize));

			if (naechsterZustand == Zustand.Ziel) {
				System.out.println("Sie haben Ihr Ziel erreicht! Anzahl Schritte: " + anzahlSchritte);
			} else {
				System.out.println("Sie sind im Wasser gelandet. Anzahl Schritte bis dahin: " + anzahlSchritte);
			}

		} catch (Exception ex) {
			System.err.println("Exception nach " + anzahlSchritte + " Schritten!");
			ex.printStackTrace();
		}
	}

	public static void printSeeColored(See see) throws InterruptedException {
		int size = see.getGroesse();
		Koordinate player = see.spielerPosition();
		for (int zeile = 0; zeile < size; ++zeile) {
			for (int spalte = 0; spalte < size; ++spalte) {

				if (player.getZeile() == zeile && player.getSpalte() == spalte) {
					System.out.print(ANSI_PURPLE + "P " + ANSI_RESET);
				} else {
					String s = getStateString(see.zustandAn(new Koordinate(zeile, spalte)));
					System.out.print(ANSI_BLACK_BACKGROUND + s);
				}
			}

			System.out.println();
		}
	}

	public static String getStateString(Zustand z) {
		if (z == Zustand.Wasser)
			return ANSI_BLUE + "W " + ANSI_RESET;
		if (z == Zustand.Eis)
			return ANSI_CYAN + "E " + ANSI_RESET;
		if (z == Zustand.Start)
			return ANSI_RED + "S " + ANSI_RESET;
		if (z == Zustand.Ziel)
			return ANSI_YELLOW + "Z " + ANSI_RESET;

		return ANSI_BLACK_BACKGROUND + ANSI_BLACK + "X " + ANSI_RESET;
	}

}
