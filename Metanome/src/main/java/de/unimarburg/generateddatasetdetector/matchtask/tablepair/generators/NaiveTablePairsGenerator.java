package de.unimarburg.generateddatasetdetector.matchtask.tablepair.generators;

import de.unimarburg.generateddatasetdetector.data.Database;
import de.unimarburg.generateddatasetdetector.data.Scenario;
import de.unimarburg.generateddatasetdetector.data.Table;
import de.unimarburg.generateddatasetdetector.matchtask.tablepair.TablePair;
import lombok.NoArgsConstructor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

@NoArgsConstructor
public class NaiveTablePairsGenerator implements TablePairsGenerator {
    private static final Logger log = LogManager.getLogger(NaiveTablePairsGenerator.class);

    @Override
    public List<TablePair> generateCandidates(Scenario scenario) {
        Database sourceDatabase = scenario.getSourceDatabase();
        Database targetDatabase = scenario.getTargetDatabase();

        List<TablePair> tablePairs = new ArrayList<>();
        for (Table sourceTable : sourceDatabase.getTables()) {
            for (Table targetTable : targetDatabase.getTables()) {
                tablePairs.add(new TablePair(sourceTable, targetTable));
            }
        }
        return tablePairs;
    }
}
