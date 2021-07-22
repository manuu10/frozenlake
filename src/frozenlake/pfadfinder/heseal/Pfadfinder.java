package frozenlake.pfadfinder.heseal;

import java.text.DecimalFormat;
import java.util.*;

import frozenlake.Koordinate;
import frozenlake.Richtung;
import frozenlake.See;
import frozenlake.Zustand;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;
//import org.neuroph.core.transfer.Linear;

public class Pfadfinder implements frozenlake.pfadfinder.IPfadfinder {
	// s(x,y) -> value
	// V(s) = (1-learnrate) * V(s) + reward + diskont*learnrate*max(sequel)
	private double[][] state_value;
	// s(x,y),a(action) -> value
	// Q(s,a) = (1-learnrate) * Q(s,a) + reward + diskont*learnrate*max(sequel)
	private double[][][] q_matrix;
	private MultiLayerPerceptron mlp;
	private DataSet trainingSet;
	private static Random rnd = new Random();
	private Koordinate myPlayer;
	private Koordinate startPos;
	private int brd_size;
	// CONSTANTS
	final int epochs = 1000;
	final double REWARD_NEUTRAL = -0.00000001;
	final double REWARD_LOOSE = -1;
	final double REWARD_WIN = 0.5;

	final double maxError = 0.00001;
	final double learnrate = 0.5;
	final double diskont = 0.999;

	boolean useStateValue = false;
	boolean useNeuralNet = false;
	boolean useOnPolicy = false;

	public Pfadfinder() {

	}

	private void initLookupTables() {
		state_value = new double[brd_size][brd_size];
		q_matrix = new double[brd_size][brd_size][4];
		int input_size = brd_size * brd_size;
		int output_size = 1;
		int hidden_size = input_size * (2 / 3) + output_size;
		mlp = new MultiLayerPerceptron(TransferFunctionType.TANH, input_size, input_size, input_size, output_size);
		trainingSet = new DataSet(input_size, output_size);
		// mlp.getLearningRule().setLearningRate(0.05);
		// mlp.getLearningRule().setMaxError(maxError);
		// mlp.getLearningRule().setMaxIterations(99);
		System.out.println("finished initialize");
	}

	@Override
	public String meinName() {
		return "eins krasser frozen laker";
	}

	@Override
	public boolean lerneSee(See see, boolean stateValue, boolean neuronalesNetz, boolean onPolicy) {
		myPlayer = new Koordinate(see.spielerPosition().getZeile(), see.spielerPosition().getSpalte());
		brd_size = see.getGroesse();
		startPos = see.spielerPosition();
		initLookupTables();
		if (stateValue && !neuronalesNetz) {
			learnStateValue(see, onPolicy);
			return true;
		}
		if (stateValue && neuronalesNetz) {
			learnNeuralNet(see, onPolicy);
			return true;
		}
		if (!stateValue && !neuronalesNetz) {
			learnQValue(see, onPolicy);
		}

		return false;
	}

	@Override
	public boolean starteUeberquerung(See see, boolean stateValue, boolean neuronalesNetz, boolean onPolicy) {
		// initialize values
		useStateValue = stateValue;
		useNeuralNet = neuronalesNetz;
		useOnPolicy = onPolicy;
		brd_size = see.getGroesse();
		myPlayer = see.spielerPosition();
		if (q_matrix == null || state_value == null || mlp == null)
			initLookupTables();

		if (useStateValue && !useNeuralNet)
			printStateValueRL();
		if (useStateValue && useNeuralNet)
			printStateValue();

		return true;
	}

	@Override
	public Richtung naechsterSchritt(Zustand ausgangszustand) {
		if (!useStateValue) {
			var stateAction = getMaxValueQMatrixPos(myPlayer, useNeuralNet);
			myPlayer = addDirToPos(myPlayer, stateAction.first);
			return stateAction.first;
		}
		Tuple<Richtung, Koordinate> direction = getMaxSequelPos(myPlayer, useNeuralNet);
		myPlayer = direction.second;
		return direction.first;
	}

	@Override
	public void versuchZuende(Zustand endzustand) {
		trainingSet = new DataSet(brd_size * brd_size, 1);
		myPlayer = startPos;
	}

	private void learnNeuralNet(See see, boolean onPolicy) {
		int currentEpoch = 0;
		int movesInEpoch = 0;
		printStateValue();
		System.out.println("using neural network to train");
		while (currentEpoch < epochs) {
			ArrayList<Tuple<Richtung, Koordinate>> possible = getValidDirections(myPlayer);
			for (Tuple<Richtung, Koordinate> tuple : possible) {
				Koordinate coord = tuple.second;
				double newValue = stateValue(coord, getReward(see.zustandAn(coord)), true);
				if (newValue <= -1)
					newValue = -0.999999999999999;
				if (newValue >= 1)
					newValue = 0.999999999999999;
				trainingSet.add(new DataSetRow(createInputVector(coord), new double[] { newValue }));
			}

			if (!onPolicy) {
				int index = rnd.nextInt(possible.size());
				myPlayer = possible.get(index).second;
			} else {
				myPlayer = getMaxSequelPos(possible, true).second;
			}

			movesInEpoch++;
			learnNN(trainingSet);
			// trainingSet = new DataSet(brd_size * brd_size, 1);
			if (see.zustandAn(myPlayer) == Zustand.Wasser || see.zustandAn(myPlayer) == Zustand.UWasser
					|| see.zustandAn(myPlayer) == Zustand.Ziel || movesInEpoch > maxNumberMovesPerEpoch()) {
				currentEpoch++;
				movesInEpoch = 0;
				versuchZuende(see.zustandAn(myPlayer));// reset players position to the start
				// if (currentEpoch % 50 == 0)
				System.out.println("\u001B[31mfinished epoch:\u001B[32m" + currentEpoch + "\u001B[0m");
			}
		}
		printStateValue();

	}

	private void learnStateValue(See see, boolean onPolicy) {
		int currentEpoch = 0;
		int movesInEpoch = 0;
		printStateValueRL();
		while (currentEpoch < epochs) {
			// update state value table
			ArrayList<Tuple<Richtung, Koordinate>> possible = getValidDirections(myPlayer);
			for (Tuple<Richtung, Koordinate> tuple : possible) {
				Koordinate coord = tuple.second;
				double newValue = stateValue(coord, getReward(see.zustandAn(coord)), false);
				state_value[coord.getZeile()][coord.getSpalte()] = newValue;
			}

			// decide wether to use off or on policy
			// off-policy -> choose a random next position
			// on-policy -> choose the position with the currently highest value;
			if (!onPolicy) {
				int index = rnd.nextInt(possible.size());
				myPlayer = possible.get(index).second;
			} else {
				myPlayer = getMaxSequelPos(possible, false).second;
			}
			movesInEpoch++;

			// stop epoch and start over when:
			// - player got stuck in water
			// - player finished the maze
			// - number of moves in current epoch is bigger than allowed
			if (see.zustandAn(myPlayer) == Zustand.Wasser || see.zustandAn(myPlayer) == Zustand.UWasser
					|| see.zustandAn(myPlayer) == Zustand.Ziel || movesInEpoch > maxNumberMovesPerEpoch()) {
				currentEpoch++;
				movesInEpoch = 0;
				versuchZuende(see.zustandAn(myPlayer));// reset players position to the start
				System.out.println("\u001B[31mfinished epoch:\u001B[32m" + currentEpoch + "\u001B[0m");
			}
		}
		printStateValueRL();
	}

	private void learnQValue(See see, boolean onPolicy) {
		int currentEpoch = 0;
		int movesInEpoch = 0;
		System.out.println("using QMatrix to learn the lake");
		while (currentEpoch < epochs) {
			// update state value table
			var possible = getValidDirectionsOnly(myPlayer);
			for (var t : possible) {
				var resultsInPosition = addDirToPos(myPlayer, t);
				var stateAction = new Tuple<Richtung, Koordinate>(t, myPlayer);
				double newValue = qValue(stateAction, getReward(see.zustandAn(resultsInPosition)), false);
				q_matrix[myPlayer.getZeile()][myPlayer.getSpalte()][t.ordinal()] = newValue;
			}

			if (!onPolicy) {
				int index = rnd.nextInt(possible.size());
				var dir = possible.get(index);
				var resultsInPosition = addDirToPos(myPlayer, dir);
				myPlayer = resultsInPosition;
			} else {
				var stateAction = getMaxValueQMatrixPos(myPlayer, false);
				myPlayer = addDirToPos(myPlayer, stateAction.first);// move to the best position
			}
			movesInEpoch++;
			if (see.zustandAn(myPlayer) == Zustand.Wasser || see.zustandAn(myPlayer) == Zustand.UWasser
					|| see.zustandAn(myPlayer) == Zustand.Ziel || movesInEpoch > maxNumberMovesPerEpoch()) {

				currentEpoch++;
				movesInEpoch = 0;
				versuchZuende(see.zustandAn(myPlayer));// reset players position to the start
				if (currentEpoch % 30 == 0)
					System.out.println("\u001B[31mfinished epoch:\u001B[32m" + currentEpoch + "\u001B[0m");
			}
		}
	}
	// ****************************************************************
	// HELPER FUNCTIONS
	// ****************************************************************

	private ArrayList<Tuple<Richtung, Koordinate>> getValidDirections(Koordinate player) {
		ArrayList<Tuple<Richtung, Koordinate>> valid = new ArrayList<Tuple<Richtung, Koordinate>>();

		for (Richtung r : Richtung.values()) {
			Koordinate new_pos = new Koordinate(player.getZeile() + r.deltaZ(), player.getSpalte() + r.deltaS());
			if (isInBoard(brd_size, new_pos))
				valid.add(new Tuple<Richtung, Koordinate>(r, new_pos));
		}
		return valid;
	}

	private ArrayList<Richtung> getValidDirectionsOnly(Koordinate player) {
		ArrayList<Richtung> valid = new ArrayList<Richtung>();

		for (Richtung r : Richtung.values()) {
			Koordinate new_pos = addDirToPos(player, r);
			if (isInBoard(brd_size, new_pos))
				valid.add(r);
		}
		return valid;
	}

	private Koordinate addDirToPos(Koordinate player, Richtung r) {
		return new Koordinate(player.getZeile() + r.deltaZ(), player.getSpalte() + r.deltaS());
	}

	private boolean isInBoard(int size, Koordinate position) {
		int x = position.getSpalte();
		int y = position.getZeile();
		return x < size && y < size && x >= 0 && y >= 0;
	}

	// calculate the state value function for a specific position
	private double stateValue(Koordinate player, double reward, boolean neuralnet) {
		double current = state_value[player.getZeile()][player.getSpalte()];

		double max_sequel = getMaxSequel(player, neuralnet);
		return (1 - learnrate) * current + reward * diskont + diskont * learnrate * max_sequel;
	}

	private double getMaxSequel(Koordinate player, boolean neuralnet) {
		Tuple<Richtung, Koordinate> t = getMaxSequelPos(player, neuralnet);
		if (neuralnet) {
			return getStateValueNN(t.second);
		}
		return state_value[t.second.getZeile()][t.second.getSpalte()];
	}

	private Tuple<Richtung, Koordinate> getMaxSequelPos(Koordinate player, boolean neuralnet) {
		ArrayList<Tuple<Richtung, Koordinate>> possible = getValidDirections(player);
		return getMaxSequelPos(possible, neuralnet);
	}

	private Tuple<Richtung, Koordinate> getMaxSequelPos(ArrayList<Tuple<Richtung, Koordinate>> possible,
			boolean neuralnet) {
		double max = Integer.MIN_VALUE;
		int index = 0;
		for (int i = 0; i < possible.size(); i++) {
			Tuple<Richtung, Koordinate> t = possible.get(i);
			double value = state_value[t.second.getZeile()][t.second.getSpalte()];
			if (neuralnet)
				value = getStateValueNN(t.second);
			if (value > max) {
				index = i;
				max = value;
			}
		}

		return possible.get(index);
	}

	private double qValue(Tuple<Richtung, Koordinate> stateAction, double reward, boolean neuralnet) {
		Koordinate pos = stateAction.second;
		Richtung r = stateAction.first;
		double current = q_matrix[pos.getZeile()][pos.getSpalte()][r.ordinal()];
		var nextPos = addDirToPos(pos, r);
		double max_sequel = getMaxValueQMatrix(nextPos, neuralnet);
		return (1 - learnrate) * current + reward * diskont + diskont * learnrate * max_sequel;
	}

	private double getMaxValueQMatrix(Koordinate player, boolean neuralnet) {
		Tuple<Richtung, Koordinate> stateAction = getMaxValueQMatrixPos(player, neuralnet);
		Koordinate pos = stateAction.second;
		Richtung r = stateAction.first;
		return q_matrix[pos.getZeile()][pos.getSpalte()][r.ordinal()];
	}

	private Tuple<Richtung, Koordinate> getMaxValueQMatrixPos(Koordinate player, boolean neuralnet) {
		ArrayList<Richtung> possible = getValidDirectionsOnly(player);
		return getMaxValueQMatrixPos(player, possible, neuralnet);
	}

	private Tuple<Richtung, Koordinate> getMaxValueQMatrixPos(Koordinate player, ArrayList<Richtung> possible,
			boolean neuralnet) {
		double max = Integer.MIN_VALUE;
		int index = 0;
		Koordinate pos = player;
		for (int i = 0; i < possible.size(); i++) {
			var dir = possible.get(i);
			double value = q_matrix[pos.getZeile()][pos.getSpalte()][dir.ordinal()];
			if (value > max) {
				index = i;
				max = value;
			}
		}

		return new Tuple<Richtung, Koordinate>(possible.get(index), player);
	}

	private double getReward(Zustand state) {
		if (state == Zustand.Wasser || state == Zustand.UWasser)
			return REWARD_LOOSE;
		if (state == Zustand.Ziel)
			return REWARD_WIN;

		return REWARD_NEUTRAL;
	}

	private double getStateValueNN(Koordinate player) {
		mlp.setInput(createInputVector(player));
		mlp.calculate();
		return mlp.getOutput()[0];
	}

	private double[] createInputVector(Koordinate player) {
		double[] input = new double[brd_size * brd_size];
		input[convert2DTo1D(player)] = 1;
		return input;
	}

	private void learnNN(DataSet set) {
		mlp.learn(set);
	}

	private int maxNumberMovesPerEpoch() {
		return brd_size * 3;
	}

	private int convert2DTo1D(Koordinate pos) {
		return brd_size * pos.getSpalte() + pos.getZeile();
	}

	public static double mapRange(double a1, double a2, double b1, double b2, double s) {
		return b1 + ((s - a1) * (b2 - b1)) / (a2 - a1);
	}

	// *******************************
	// PRINT FUNCTIONS
	// *******************************
	private static DecimalFormat df = new DecimalFormat("#.#####");

	private void printStateValue() {
		System.out.println("StateValues");
		for (int i = 0; i < brd_size; i++) {
			for (int j = 0; j < brd_size; j++) {
				System.out.print(df.format(getStateValueNN(new Koordinate(i, j))) + " ");
			}
			System.out.println();
		}
	}

	private void printStateValueRL() {
		System.out.println("StateValues");
		for (int i = 0; i < brd_size; i++) {
			for (int j = 0; j < brd_size; j++) {
				double value = state_value[i][j];
				System.out.print(df.format(value) + " ");
			}
			System.out.println();
		}

	}
}
