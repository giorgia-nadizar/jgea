/*
 * Copyright 2022 eric
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.malelab.jgea.tui;

import com.googlecode.lanterna.TerminalSize;
import com.googlecode.lanterna.TextColor;
import com.googlecode.lanterna.graphics.TextGraphics;
import com.googlecode.lanterna.input.KeyStroke;
import com.googlecode.lanterna.input.KeyType;
import com.googlecode.lanterna.screen.Screen;
import com.googlecode.lanterna.terminal.DefaultTerminalFactory;
import it.units.malelab.jgea.core.listener.*;
import it.units.malelab.jgea.core.solver.state.State;
import it.units.malelab.jgea.core.util.*;
import it.units.malelab.jgea.tui.util.Point;
import it.units.malelab.jgea.tui.util.Rectangle;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.logging.*;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.tui.util.DrawUtils.*;

/**
 * @author "Eric Medvet" on 2022/09/02 for jgea
 */
public class TerminalMonitor<E, K> extends Handler implements ListenerFactory<E, K>, ProgressMonitor {

  private final static Configuration DEFAULT_CONFIGURATION = new Configuration(
      0.7f,
      0.8f,
      0.5f,
      0.65f,
      16,
      250,
      true,
      true
  );

  private final static TextColor FRAME_COLOR = TextColor.Factory.fromString("#105010");
  private final static TextColor FRAME_LABEL_COLOR = TextColor.Factory.fromString("#10A010");
  private final static TextColor DATA_LABEL_COLOR = TextColor.Factory.fromString("#A01010");
  private final static TextColor MAIN_DATA_COLOR = TextColor.Factory.fromString("#F0F0F0");
  private final static TextColor DATA_COLOR = TextColor.Factory.fromString("#A0A0A0");
  private final static TextColor PLOT_BG_COLOR = TextColor.Factory.fromString("#101010");
  private final static TextColor PLOT1_COLOR = TextColor.Factory.fromString("#FF1010");
  private final static TextColor PLOT2_COLOR = TextColor.Factory.fromString("#105010");

  private final static String LEVEL_FORMAT = "%4.4s";
  private final static String DATETIME_FORMAT = "%1$tm-%1$td %1$tH:%1$tM:%1$tS";

  private final static int LOG_HISTORY_SIZE = 100;
  private final static int RUN_HISTORY_SIZE = 1000;

  private final static Map<Level, TextColor> LEVEL_COLORS = Map.ofEntries(
      Map.entry(
          Level.SEVERE,
          TextColor.Factory.fromString("#EE3E38")
      ),
      Map.entry(Level.WARNING, TextColor.Factory.fromString("#FBA465")),
      Map.entry(Level.INFO, TextColor.Factory.fromString("#D8E46B")),
      Map.entry(Level.CONFIG, TextColor.Factory.fromString("#6D8700"))
  );
  private final static Logger L = Logger.getLogger(TerminalMonitor.class.getName());
  private final List<? extends NamedFunction<? super E, ?>> eFunctions;
  private final List<? extends NamedFunction<? super K, ?>> kFunctions;
  private final List<PlotTableBuilder<? super E>> plotTableBuilders;
  private final List<String> formats;
  private final Configuration configuration;
  private final ScheduledFuture<?> painterTask;
  private final List<LogRecord> logRecords;
  private final Table<Object> runTable;
  private final Instant startingInstant;
  private final List<Handler> originalHandlers;

  private final List<Accumulator<? super E, Table<Number>>> plotAccumulators;
  private final ScheduledExecutorService uiExecutorService;
  private Screen screen;
  private Progress overallProgress;
  private Progress partialProgress;
  private String lastProgressMessage;
  private Instant lastProgressInstant;


  public TerminalMonitor(
      List<NamedFunction<? super E, ?>> eFunctions,
      List<NamedFunction<? super K, ?>> kFunctions,
      List<PlotTableBuilder<? super E>> plotTableBuilders
  ) {
    this(eFunctions, kFunctions, plotTableBuilders, DEFAULT_CONFIGURATION);
  }

  public TerminalMonitor(
      List<NamedFunction<? super E, ?>> eFunctions,
      List<NamedFunction<? super K, ?>> kFunctions,
      List<PlotTableBuilder<? super E>> plotTableBuilders,
      Configuration configuration
  ) {
    //set functions
    this.eFunctions = configuration.robust() ? eFunctions.stream().map(NamedFunction::robust).toList() : eFunctions;
    this.kFunctions = configuration.robust() ? kFunctions.stream().map(NamedFunction::robust).toList() : kFunctions;
    this.plotTableBuilders = plotTableBuilders;
    formats = Misc.concat(List.of(
        eFunctions.stream().map(NamedFunction::getFormat).toList(),
        kFunctions.stream().map(NamedFunction::getFormat).toList()
    ));
    plotAccumulators = new ArrayList<>();
    //read configuration
    this.configuration = configuration;
    //prepare data object stores
    logRecords = new LinkedList<>();
    runTable = new ArrayTable<>(Misc.concat(List.of(
        eFunctions.stream().map(NamedFunction::getName).toList(),
        kFunctions.stream().map(NamedFunction::getName).toList()
    )));
    //prepare screen
    DefaultTerminalFactory defaultTerminalFactory = new DefaultTerminalFactory();
    try {
      screen = defaultTerminalFactory.createScreen();
      screen.startScreen();
    } catch (IOException e) {
      L.severe(String.format("Cannot create or start screen: %s", e));
    }
    if (screen != null) {
      screen.setCursorPosition(null);
      repaint();
    }
    //start painting scheduler
    uiExecutorService = Executors.newSingleThreadScheduledExecutor();
    painterTask = uiExecutorService.scheduleAtFixedRate(
        this::repaint,
        0,
        configuration.refreshIntervalMillis,
        TimeUnit.MILLISECONDS
    );
    //capture logs
    Logger mainLogger = Logger.getLogger("");
    mainLogger.setLevel(Level.CONFIG);
    mainLogger.addHandler(this);
    originalHandlers = Arrays.stream(mainLogger.getHandlers()).filter(h -> h instanceof ConsoleHandler).toList();
    originalHandlers.forEach(mainLogger::removeHandler);
    //set default locale
    Locale.setDefault(Locale.ENGLISH);
    startingInstant = Instant.now();
  }

  public record Configuration(
      float verticalSplit, float leftHorizontalSplit, float rightHorizontalSplit, float plotHorizontalSplit,
      int barLength, int refreshIntervalMillis, boolean dumpLogAfterStop, boolean robust
  ) {}

  @Override
  public Listener<E> build(K k) {
    List<?> kItems = kFunctions.stream().map(f -> f.apply(k)).toList();
    List<? extends Accumulator<? super E, Table<Number>>> localPlotAccumulators = plotTableBuilders.stream()
        .map(b -> b.build(null))
        .toList();
    synchronized (runTable) {
      runTable.clear();
      plotAccumulators.clear();
      plotAccumulators.addAll(localPlotAccumulators);
    }
    return e -> {
      List<?> eItems = eFunctions.stream().map(f -> f.apply(e)).toList();
      List<Object> row = Misc.concat(List.of(eItems, kItems));
      synchronized (runTable) {
        runTable.addRow(row);
        while (runTable.nRows() > RUN_HISTORY_SIZE) {
          runTable.removeRow(0);
        }
        localPlotAccumulators.forEach(a -> a.listen(e));
      }
      if (e instanceof State state) {
        partialProgress = state.getProgress();
      }
    };
  }

  @Override
  public void shutdown() {
    stop();
  }

  private double getCPULoad() {
    return ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class).getSystemLoadAverage();
  }

  private int getNumberOfProcessors() {
    return ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class).getAvailableProcessors();
  }

  @Override
  public void notify(Progress progress, String message) {
    overallProgress = progress;
    lastProgressMessage = message;
    lastProgressInstant = Instant.now();
  }

  @Override
  public synchronized void publish(LogRecord record) {
    synchronized (logRecords) {
      logRecords.add(record);
      while (logRecords.size() > LOG_HISTORY_SIZE) {
        logRecords.remove(0);
      }
    }
  }

  @Override
  public void flush() {
  }

  @Override
  public void close() throws SecurityException {
  }

  private void repaint() {
    if (screen == null) {
      return;
    }
    //check keystrokes
    try {
      KeyStroke k = screen.pollInput();
      if (k != null && ((k.getCharacter().equals('c') && k.isCtrlDown()) || k.getKeyType().equals(KeyType.EOF))) {
        stop();
      }
    } catch (IOException e) {
      L.warning(String.format("Cannot check key strokes: %s", e));
    }
    //update size
    TerminalSize size = screen.doResizeIfNecessary();
    TextGraphics tg = screen.newTextGraphics();
    Rectangle r;
    if (size == null) {
      size = screen.getTerminalSize();
    } else {
      screen.clear();
    }
    //adjust rectangles
    Rectangle all = new Rectangle(new Point(0, 0), new Point(size.getColumns(), size.getRows()));
    Rectangle e = all.splitHorizontally(configuration.verticalSplit).get(0);
    Rectangle w = all.splitHorizontally(configuration.verticalSplit).get(1);
    Rectangle runR = e.splitVertically(configuration.leftHorizontalSplit).get(0);
    Rectangle logR = e.splitVertically(configuration.leftHorizontalSplit).get(1);
    Rectangle legendR = w.splitVertically(configuration.rightHorizontalSplit).get(0);
    Rectangle statusR = w.splitVertically(configuration.rightHorizontalSplit).get(1);
    List<Rectangle> plotRs;
    if (plotTableBuilders.isEmpty()) {
      plotRs = List.of();
    } else {
      Rectangle plotsR = runR.splitVertically(configuration.plotHorizontalSplit).get(1);
      runR = runR.splitVertically(configuration.plotHorizontalSplit).get(0);
      if (plotTableBuilders.size() > 1) {
        float[] splits = new float[plotTableBuilders.size() - 1];
        splits[0] = 1f / (float) plotTableBuilders.size();
        for (int i = 1; i < splits.length; i++) {
          splits[i] = splits[i - 1] + 1f / (float) plotTableBuilders.size();
        }
        plotRs = plotsR.splitHorizontally(splits);
      } else {
        plotRs = List.of(plotsR);
      }
    }
    //draw structure
    drawFrame(tg, runR, "Ongoing run", FRAME_COLOR, FRAME_LABEL_COLOR);
    drawFrame(tg, legendR, "Legend", FRAME_COLOR, FRAME_LABEL_COLOR);
    drawFrame(tg, logR, "Log", FRAME_COLOR, FRAME_LABEL_COLOR);
    drawFrame(tg, statusR, "Status", FRAME_COLOR, FRAME_LABEL_COLOR);
    for (int i = 0; i < plotTableBuilders.size(); i++) {
      String plotName = plotTableBuilders.get(i).yNames().get(0) + " vs. " + plotTableBuilders.get(i).xName();
      drawFrame(tg, plotRs.get(i), plotName, FRAME_COLOR, FRAME_LABEL_COLOR);
    }
    //draw data: logs
    int levelW = String.format(LEVEL_FORMAT, Level.WARNING).length();
    int dateW = String.format(DATETIME_FORMAT, Instant.now().getEpochSecond()).length();
    r = logR.inner(1);
    clear(tg, r);
    synchronized (logRecords) {
      for (int i = 0; i < Math.min(r.h(), logRecords.size()); i = i + 1) {
        LogRecord record = logRecords.get(logRecords.size() - 1 - i);
        tg.setForegroundColor(LEVEL_COLORS.getOrDefault(record.getLevel(), DATA_COLOR));
        clipPut(tg, r, 0, i, String.format(LEVEL_FORMAT, record.getLevel()));
        tg.setForegroundColor(MAIN_DATA_COLOR);
        clipPut(tg, r, levelW + 1, i, String.format(DATETIME_FORMAT, record.getMillis()));
        tg.setForegroundColor(DATA_COLOR);
        clipPut(tg, r, levelW + 1 + dateW + 1, i, record.getMessage());
      }
    }
    //draw data: status
    r = statusR.inner(1);
    clear(tg, r);
    tg.setForegroundColor(DATA_LABEL_COLOR);
    clipPut(tg, r, 0, 0, "Machine:");
    clipPut(tg, r, 0, 1, "Loc time:");
    clipPut(tg, r, 0, 2, "CPU load:");
    clipPut(tg, r, 0, 3, "Memory:");
    clipPut(tg, r, 0, 4, "Over. progr.:");
    clipPut(tg, r, 0, 5, "Curr. progr.:");
    clipPut(tg, r, 0, 6, "ETA:");
    final int labelLength = "Over. progr.:".length();
    clipPut(tg, r, 0, 7, "Last progress message:");
    tg.setForegroundColor(MAIN_DATA_COLOR);
    clipPut(tg, r, 14, 0, StringUtils.getMachineName());
    clipPut(tg, r, 14, 1, String.format(DATETIME_FORMAT, Date.from(Instant.now())));
    float maxGigaMemory = Runtime.getRuntime().maxMemory() / 1024f / 1024f / 1024f;
    float usedGigaMemory = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime()
        .freeMemory()) / 1024f / 1024f / 1024f;
    double cpuLoad = getCPULoad();
    int nOfProcessors = getNumberOfProcessors();
    drawHorizontalBar(
        tg,
        r,
        labelLength + 1,
        2,
        cpuLoad,
        0,
        nOfProcessors,
        configuration.barLength,
        PLOT1_COLOR,
        PLOT_BG_COLOR
    );
    clipPut(
        tg,
        r,
        labelLength + configuration.barLength + 2,
        2,
        String.format("%.2f on %d cores", cpuLoad, 2 * nOfProcessors)
    );
    drawHorizontalBar(
        tg,
        r,
        labelLength + 1,
        3,
        usedGigaMemory,
        0,
        maxGigaMemory,
        configuration.barLength,
        PLOT1_COLOR,
        PLOT_BG_COLOR
    );
    clipPut(tg, r, labelLength + configuration.barLength + 2, 3, String.format("%.1fGB", maxGigaMemory));
    if (overallProgress != null) {
      Progress progress = new Progress(overallProgress.start(), overallProgress.end(), overallProgress.current());
      if (partialProgress != null && !Double.isNaN(partialProgress.rate())) {
        progress = new Progress(
            progress.start(),
            progress.end(),
            Math.floor(progress.current().doubleValue()) + partialProgress.rate()
        );
      }
      drawHorizontalBar(
          tg,
          r,
          labelLength + 1,
          4,
          progress.rate(),
          0,
          1,
          configuration.barLength,
          PLOT1_COLOR,
          PLOT_BG_COLOR
      );
      clipPut(tg, r, labelLength + configuration.barLength + 2, 4, "%3.0f%%".formatted(progress.rate() * 100));
      if (lastProgressInstant != null) {
        if (progress.rate() > 0) {
          Instant eta = startingInstant.plus(Math.round(ChronoUnit.MILLIS.between(
              startingInstant,
              lastProgressInstant
          ) / progress.rate()), ChronoUnit.MILLIS);
          clipPut(tg, r, labelLength + 1, 6, DATETIME_FORMAT.formatted(Date.from(eta)));
        }
      }
    }
    if (partialProgress != null) {
      drawHorizontalBar(
          tg,
          r,
          labelLength + 1,
          5,
          partialProgress.rate(),
          0,
          1,
          configuration.barLength,
          PLOT1_COLOR,
          PLOT_BG_COLOR
      );
      clipPut(tg, r, labelLength + configuration.barLength + 2, 5, "%3.0f%%".formatted(partialProgress.rate() * 100));
    }
    if (lastProgressMessage != null) {
      tg.setForegroundColor(DATA_COLOR);
      clipPut(tg, r, 0, 8, lastProgressMessage);
    }
    //draw data: legend
    synchronized (runTable) {
      r = legendR.inner(1);
      clear(tg, r);
      List<Pair<String, String>> legendItems = runTable.names().stream().map(s -> new Pair<>(
          StringUtils.collapse(s),
          s
      )).toList();
      int shortLabelW = legendItems.stream().mapToInt(p -> p.first().length()).max().orElse(0);
      for (int i = 0; i < legendItems.size(); i = i + 1) {
        tg.setForegroundColor(DATA_LABEL_COLOR);
        clipPut(tg, r, 0, i, legendItems.get(i).first());
        tg.setForegroundColor(DATA_COLOR);
        clipPut(tg, r, shortLabelW + 1, i, legendItems.get(i).second());
      }
      //draw data: run
      r = runR.inner(1);
      clear(tg, r);
      int[] colWidths = IntStream.range(0, runTable.nColumns()).map(x -> Math.max(
          legendItems.get(x).first().length(),
          runTable.column(x).stream().mapToInt(o -> String.format(formats.get(x), o).length()).max().orElse(0)
      )).toArray();
      tg.setForegroundColor(DATA_LABEL_COLOR);
      int x = 0;
      for (int i = 0; i < colWidths.length; i = i + 1) {
        clipPut(tg, r, x, 0, legendItems.get(i).first());
        x = x + colWidths[i] + 1;
      }
      tg.setForegroundColor(DATA_COLOR);
      for (int j = 0; j < Math.min(runTable.nRows(), r.h()); j = j + 1) {
        x = 0;
        int rowIndex = runTable.nRows() - j - 1;
        for (int i = 0; i < colWidths.length; i = i + 1) {
          Object value = runTable.get(i, rowIndex);
          try {
            clipPut(tg, r, x, j + 1, String.format(formats.get(i), value));
          } catch (IllegalFormatConversionException ex) {
            L.warning(String.format(
                "Cannot format %s %s as a \"%s\" with %s: %s",
                value.getClass().getSimpleName(),
                value,
                formats.get(rowIndex),
                legendItems.get(i).second(),
                ex
            ));
          }
          x = x + colWidths[i] + 1;
        }
      }
    }
    //draw plots
    if (!plotAccumulators.isEmpty()) {
      for (int i = 0; i < plotAccumulators.size(); i = i + 1) {
        double minX = Double.NaN;
        double maxX = Double.NaN;
        double minY = Double.NaN;
        double maxY = Double.NaN;
        if (plotTableBuilders.get(i) instanceof XYPlotTableBuilder<?> xyPlotTableBuilder) {
          minX = Double.isFinite(xyPlotTableBuilder.getMinX()) ? xyPlotTableBuilder.getMinX() : Double.NaN;
          maxX = Double.isFinite(xyPlotTableBuilder.getMaxX()) ? xyPlotTableBuilder.getMaxX() : Double.NaN;
          minY = Double.isFinite(xyPlotTableBuilder.getMinY()) ? xyPlotTableBuilder.getMinY() : Double.NaN;
          maxY = Double.isFinite(xyPlotTableBuilder.getMaxY()) ? xyPlotTableBuilder.getMaxY() : Double.NaN;
        }
        try {
          Table<Number> table = plotAccumulators.get(i).get().filter(row -> row.stream()
              .noneMatch(n -> Double.isNaN(n.doubleValue())));
          drawPlot(
              tg,
              plotRs.get(i).inner(1),
              table,
              PLOT2_COLOR,
              MAIN_DATA_COLOR,
              PLOT_BG_COLOR,
              plotTableBuilders.get(i).xFormat(),
              plotTableBuilders.get(i).yFormats().get(0),
              minX,
              maxX,
              minY,
              maxY
          );
        } catch (RuntimeException ex) {
          L.warning(String.format(
              "Cannot do plot %s vs. %s: %s",
              plotTableBuilders.get(i).yNames().get(0),
              plotTableBuilders.get(i).xName(),
              ex
          ));
        }
      }
    }
    //refresh
    try {
      screen.refresh();
    } catch (IOException ex) {
      L.warning(String.format("Cannot refresh screen: %s", ex));
    }
  }

  private void stop() {
    try {
      screen.stopScreen();
    } catch (IOException e) {
      L.warning(String.format("Cannot stop screen: %s", e));
    }
    painterTask.cancel(false);
    L.info("Closed");
    Logger.getLogger("").removeHandler(this);
    originalHandlers.forEach(h -> Logger.getLogger("").addHandler(h));
    if (configuration.dumpLogAfterStop()) {
      logRecords.forEach(L::log);
    }
    uiExecutorService.shutdownNow();
  }

}
