package frozenlake.pfadfinder;

import frozenlake.Richtung;
import frozenlake.See;
import frozenlake.Zustand;

public interface IPfadfinder {
	
	/** Gibt den Namen Ihres Pfadfinders (=Gruppenname) zur�ck */
	public String meinName();
	
	/** Startet das best�rkende Lernen f�r einen See. Sie k�nnen das Netzwerk nach dem
	 * Lernvorgang speichern, damit es anschlie�end bei starteUeberquerung verwendet
	 * werden kann
	 * @param see Zu �berquerender See
	 * @param stateValue False: QValues verwenden, True: StateValue verwenden
 	 * @param neuronalesNetz True, wenn ein neuronales Netz verwendet wird
	 * @param onPolicy False: Zuf�llige Z�ge beim Lernen, true: onPolicy-Lernen
	 * @return true, wenn Lernvorgang erfolgreich war
	 */
	public boolean lerneSee(See see, boolean stateValue, boolean neuronalesNetz, boolean onPolicy);

	/** Startet den �berquerungsvorgang f�r einen See. Sie sollten den Trainingsvorgang 
	 * vorher durchgef�hrt haben. Wenn vorhanden, nutzen Sie das abgespeicherte Netz.
	 * @param see Zu �berquerender See
	 * @param stateValue False: QValues verwenden, True: StateValue verwenden
	 * @param neuronalesNetz True, wenn ein neuronales Netz verwendet wird
	 * @param onPolicy False: Zuf�llige Z�ge beim Lernen, true: onPolicy-Lernen
	 * @return true, wenn der �berquerungsvorgang erfolgreich initialisiert wurde.
	 */
	public boolean starteUeberquerung(See see, boolean stateValue, boolean neuronalesNetz, boolean onPolicy );
	
	/** Wird wiederholt nach Start der �berquerung aufgerufen und muss den jeweils 
	 * n�chsten Schritt liefern.
	 * @param ausgangszustand: Gibt an, was sich auf dem aktuellen Feld befindet. Kann
	 * nur "Start" oder "Eis" sein.
	 * @return Richtung des n�chsten Schrittes
	 */
	public Richtung naechsterSchritt(Zustand ausgangszustand);
	
	/** Wird aufgerufen, wenn Sie das Ziel erreicht haben (endzustand = Ziel) oder wenn
	 * Ihr IPfadfinder ins Wasster gefallen ist (endzustand = Wasser). In beiden F�llen
	 * ist die �berquerung zuende.
	 * @param endzustand
	 */
	public void versuchZuende(Zustand endzustand);
}
