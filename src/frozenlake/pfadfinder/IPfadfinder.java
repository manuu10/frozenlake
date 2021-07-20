package frozenlake.pfadfinder;

import frozenlake.Richtung;
import frozenlake.See;
import frozenlake.Zustand;

public interface IPfadfinder {
	
	/** Gibt den Namen Ihres Pfadfinders (=Gruppenname) zurück */
	public String meinName();
	
	/** Startet das bestärkende Lernen für einen See. Sie können das Netzwerk nach dem
	 * Lernvorgang speichern, damit es anschließend bei starteUeberquerung verwendet
	 * werden kann
	 * @param see Zu überquerender See
	 * @param stateValue False: QValues verwenden, True: StateValue verwenden
 	 * @param neuronalesNetz True, wenn ein neuronales Netz verwendet wird
	 * @param onPolicy False: Zufällige Züge beim Lernen, true: onPolicy-Lernen
	 * @return true, wenn Lernvorgang erfolgreich war
	 */
	public boolean lerneSee(See see, boolean stateValue, boolean neuronalesNetz, boolean onPolicy);

	/** Startet den Überquerungsvorgang für einen See. Sie sollten den Trainingsvorgang 
	 * vorher durchgeführt haben. Wenn vorhanden, nutzen Sie das abgespeicherte Netz.
	 * @param see Zu überquerender See
	 * @param stateValue False: QValues verwenden, True: StateValue verwenden
	 * @param neuronalesNetz True, wenn ein neuronales Netz verwendet wird
	 * @param onPolicy False: Zufällige Züge beim Lernen, true: onPolicy-Lernen
	 * @return true, wenn der Überquerungsvorgang erfolgreich initialisiert wurde.
	 */
	public boolean starteUeberquerung(See see, boolean stateValue, boolean neuronalesNetz, boolean onPolicy );
	
	/** Wird wiederholt nach Start der Überquerung aufgerufen und muss den jeweils 
	 * nächsten Schritt liefern.
	 * @param ausgangszustand: Gibt an, was sich auf dem aktuellen Feld befindet. Kann
	 * nur "Start" oder "Eis" sein.
	 * @return Richtung des nächsten Schrittes
	 */
	public Richtung naechsterSchritt(Zustand ausgangszustand);
	
	/** Wird aufgerufen, wenn Sie das Ziel erreicht haben (endzustand = Ziel) oder wenn
	 * Ihr IPfadfinder ins Wasster gefallen ist (endzustand = Wasser). In beiden Fällen
	 * ist die Überquerung zuende.
	 * @param endzustand
	 */
	public void versuchZuende(Zustand endzustand);
}
