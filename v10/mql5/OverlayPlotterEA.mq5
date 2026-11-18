//+------------------------------------------------------------------+
//| OverlayPlotterEA.mq5                                            |
//| Time Series AI for Real-Time Trading Risk Management             |
//| Student: Mohammad Naqi Zaidi | B00963745                         |
//|                                                                  |
//| Purpose:                                                         |
//|   Reads CSV overlay files from MQL5/Files/overlays/ and draws    |
//|   visual elements on the chart:                                  |
//|     - Support / Resistance levels                                |
//|     - Points of Interest (POI)                                   |
//|     - Fair Value Gaps (FVG)                                      |
//|     - Market Structure (swing highs/lows, BOS, CHoCH)            |
//|     - Regime labels and action states                            |
//|                                                                  |
//| File contract:                                                   |
//|   IN: MQL5/Files/overlays/support_resistance.csv                 |
//|   IN: MQL5/Files/overlays/poi.csv                                |
//|   IN: MQL5/Files/overlays/fvg.csv                                |
//|   IN: MQL5/Files/overlays/market_structure.csv                   |
//|   IN: MQL5/Files/overlays/regime_state.csv                       |
//+------------------------------------------------------------------+
#property copyright "Mohammad Naqi Zaidi"
#property link      ""
#property version   "1.00"
#property strict
#property indicator_chart_window

//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input string   InpOverlayDir    = "overlays";       // Overlay directory
input int      InpPollSeconds   = 5;                 // Refresh interval (sec)
input bool     InpShowSR        = true;              // Show Support/Resistance
input bool     InpShowPOI       = true;              // Show Points of Interest
input bool     InpShowFVG       = true;              // Show Fair Value Gaps
input bool     InpShowStructure = true;              // Show Market Structure
input bool     InpShowRegime    = true;              // Show Regime/Action state
input color    InpSupportColor  = clrDodgerBlue;     // Support level color
input color    InpResistColor   = clrOrangeRed;      // Resistance level color
input color    InpPOIColor      = clrGold;           // POI color
input color    InpFVGBullColor  = clrLimeGreen;      // Bullish FVG color
input color    InpFVGBearColor  = clrCrimson;        // Bearish FVG color
input color    InpSwingHiColor  = clrRed;            // Swing high marker
input color    InpSwingLoColor  = clrGreen;          // Swing low marker

//+------------------------------------------------------------------+
//| Global prefix for all drawn objects                              |
//+------------------------------------------------------------------+
string g_prefix = "GR_OVL_";

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   EventSetTimer(InpPollSeconds);
   Print("OverlayPlotterEA initialized. Poll=", InpPollSeconds, "s");
   RefreshOverlays();
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization — clean up all drawn objects              |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   ObjectsDeleteAll(0, g_prefix);
   Print("OverlayPlotterEA stopped. All overlays removed.");
}

//+------------------------------------------------------------------+
//| Timer event — refresh overlays                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   RefreshOverlays();
}

//+------------------------------------------------------------------+
//| Main refresh: read all CSV files and redraw                      |
//+------------------------------------------------------------------+
void RefreshOverlays()
{
   // Clear previous objects
   ObjectsDeleteAll(0, g_prefix);

   if(InpShowSR)        DrawSupportResistance();
   if(InpShowPOI)       DrawPOI();
   if(InpShowFVG)       DrawFVG();
   if(InpShowStructure) DrawMarketStructure();
   if(InpShowRegime)    DrawRegimeState();

   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| Draw Support/Resistance levels                                   |
//| CSV: ts_utc, price, type (support/resistance), strength          |
//+------------------------------------------------------------------+
void DrawSupportResistance()
{
   string filename = InpOverlayDir + "\\support_resistance.csv";
   if(!FileIsExist(filename)) return;

   int handle = FileOpen(filename, FILE_READ|FILE_CSV|FILE_SHARE_READ, ',');
   if(handle == INVALID_HANDLE) return;

   // Skip header
   SkipCSVHeader(handle, 4);

   int count = 0;
   while(!FileIsEnding(handle))
   {
      string ts    = FileReadString(handle); if(ts == "") break;
      string price = FileReadString(handle);
      string type  = FileReadString(handle);
      string str   = FileReadString(handle);

      double priceVal = StringToDouble(price);
      if(priceVal <= 0) continue;

      string name = g_prefix + "SR_" + IntegerToString(count);
      color  clr  = (type == "support") ? InpSupportColor : InpResistColor;
      int    width = (StringToInteger(str) >= 3) ? 2 : 1;
      ENUM_LINE_STYLE style = (StringToInteger(str) >= 3) ?
                               STYLE_SOLID : STYLE_DOT;

      ObjectCreate(0, name, OBJ_HLINE, 0, 0, priceVal);
      ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
      ObjectSetInteger(0, name, OBJPROP_STYLE, style);
      ObjectSetString(0, name, OBJPROP_TOOLTIP,
                      type + " @ " + price + " (str=" + str + ")");
      ObjectSetInteger(0, name, OBJPROP_BACK, true);

      count++;
   }
   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Draw Points of Interest                                          |
//| CSV: ts_utc, price, label, notes                                 |
//+------------------------------------------------------------------+
void DrawPOI()
{
   string filename = InpOverlayDir + "\\poi.csv";
   if(!FileIsExist(filename)) return;

   int handle = FileOpen(filename, FILE_READ|FILE_CSV|FILE_SHARE_READ, ',');
   if(handle == INVALID_HANDLE) return;

   SkipCSVHeader(handle, 4);

   int count = 0;
   while(!FileIsEnding(handle))
   {
      string ts    = FileReadString(handle); if(ts == "") break;
      string price = FileReadString(handle);
      string label = FileReadString(handle);
      string notes = FileReadString(handle);

      double priceVal = StringToDouble(price);
      datetime time   = ParseUTCTime(ts);
      if(priceVal <= 0 || time == 0) continue;

      string name = g_prefix + "POI_" + IntegerToString(count);

      ObjectCreate(0, name, OBJ_ARROW, 0, time, priceVal);
      ObjectSetInteger(0, name, OBJPROP_ARROWCODE, 159); // diamond
      ObjectSetInteger(0, name, OBJPROP_COLOR, InpPOIColor);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
      ObjectSetString(0, name, OBJPROP_TOOLTIP,
                      label + ": " + notes);

      count++;
   }
   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Draw Fair Value Gaps                                             |
//| CSV: ts_utc, high, low, direction (bull/bear)                    |
//+------------------------------------------------------------------+
void DrawFVG()
{
   string filename = InpOverlayDir + "\\fvg.csv";
   if(!FileIsExist(filename)) return;

   int handle = FileOpen(filename, FILE_READ|FILE_CSV|FILE_SHARE_READ, ',');
   if(handle == INVALID_HANDLE) return;

   SkipCSVHeader(handle, 4);

   int count = 0;
   while(!FileIsEnding(handle))
   {
      string ts   = FileReadString(handle); if(ts == "") break;
      string hi   = FileReadString(handle);
      string lo   = FileReadString(handle);
      string dir  = FileReadString(handle);

      double hiVal = StringToDouble(hi);
      double loVal = StringToDouble(lo);
      datetime time1 = ParseUTCTime(ts);
      if(hiVal <= 0 || loVal <= 0 || time1 == 0) continue;

      // FVG rectangle extends 10 bars forward
      datetime time2 = time1 + PeriodSeconds(PERIOD_M1) * 10;

      string name = g_prefix + "FVG_" + IntegerToString(count);
      color  clr  = (dir == "bull") ? InpFVGBullColor : InpFVGBearColor;

      ObjectCreate(0, name, OBJ_RECTANGLE, 0, time1, hiVal, time2, loVal);
      ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, name, OBJPROP_FILL, true);
      ObjectSetInteger(0, name, OBJPROP_BACK, true);
      ObjectSetString(0, name, OBJPROP_TOOLTIP,
                      dir + " FVG [" + lo + " – " + hi + "]");

      count++;
   }
   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Draw Market Structure (swing points, BOS, CHoCH)                 |
//| CSV: ts_utc, price, type (swing_hi/swing_lo/bos/choch), label    |
//+------------------------------------------------------------------+
void DrawMarketStructure()
{
   string filename = InpOverlayDir + "\\market_structure.csv";
   if(!FileIsExist(filename)) return;

   int handle = FileOpen(filename, FILE_READ|FILE_CSV|FILE_SHARE_READ, ',');
   if(handle == INVALID_HANDLE) return;

   SkipCSVHeader(handle, 4);

   int count = 0;
   while(!FileIsEnding(handle))
   {
      string ts    = FileReadString(handle); if(ts == "") break;
      string price = FileReadString(handle);
      string type  = FileReadString(handle);
      string label = FileReadString(handle);

      double priceVal = StringToDouble(price);
      datetime time   = ParseUTCTime(ts);
      if(priceVal <= 0 || time == 0) continue;

      string name = g_prefix + "MS_" + IntegerToString(count);

      if(type == "swing_hi")
      {
         ObjectCreate(0, name, OBJ_ARROW, 0, time, priceVal);
         ObjectSetInteger(0, name, OBJPROP_ARROWCODE, 234); // down arrow
         ObjectSetInteger(0, name, OBJPROP_COLOR, InpSwingHiColor);
         ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
      }
      else if(type == "swing_lo")
      {
         ObjectCreate(0, name, OBJ_ARROW, 0, time, priceVal);
         ObjectSetInteger(0, name, OBJPROP_ARROWCODE, 233); // up arrow
         ObjectSetInteger(0, name, OBJPROP_COLOR, InpSwingLoColor);
         ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
      }
      else if(type == "bos" || type == "choch")
      {
         // Draw a short horizontal line with label
         datetime time2 = time + PeriodSeconds(PERIOD_M1) * 5;
         ObjectCreate(0, name, OBJ_TREND, 0, time, priceVal, time2, priceVal);
         ObjectSetInteger(0, name, OBJPROP_COLOR,
                          (type == "bos") ? clrWhite : clrYellow);
         ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
         ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DASH);
         ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);

         // Label
         string lblName = name + "_lbl";
         ObjectCreate(0, lblName, OBJ_TEXT, 0, time, priceVal);
         ObjectSetString(0, lblName, OBJPROP_TEXT,
                         StringToUpper(type) + (label != "" ? ": " + label : ""));
         ObjectSetInteger(0, lblName, OBJPROP_COLOR,
                          (type == "bos") ? clrWhite : clrYellow);
         ObjectSetInteger(0, lblName, OBJPROP_FONTSIZE, 8);
      }

      ObjectSetString(0, name, OBJPROP_TOOLTIP, type + " @ " + price);
      count++;
   }
   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Draw Regime/Action state label on chart                          |
//| CSV: ts_utc, regime_label, anomaly_level, action, reason         |
//+------------------------------------------------------------------+
void DrawRegimeState()
{
   string filename = InpOverlayDir + "\\regime_state.csv";
   if(!FileIsExist(filename)) return;

   int handle = FileOpen(filename, FILE_READ|FILE_CSV|FILE_SHARE_READ, ',');
   if(handle == INVALID_HANDLE) return;

   SkipCSVHeader(handle, 5);

   // Read the last row only (current state)
   string lastTs      = "";
   string lastRegime  = "";
   string lastAnomaly = "";
   string lastAction  = "";
   string lastReason  = "";

   while(!FileIsEnding(handle))
   {
      string ts  = FileReadString(handle); if(ts == "") break;
      string reg = FileReadString(handle);
      string ano = FileReadString(handle);
      string act = FileReadString(handle);
      string rsn = FileReadString(handle);

      if(ts != "")
      {
         lastTs      = ts;
         lastRegime  = reg;
         lastAnomaly = ano;
         lastAction  = act;
         lastReason  = rsn;
      }
   }
   FileClose(handle);

   if(lastTs == "") return;

   // Determine colors based on action
   color bgColor, textColor;
   if(lastAction == "allow")
   {
      bgColor   = clrDarkGreen;
      textColor = clrLime;
   }
   else if(lastAction == "throttle")
   {
      bgColor   = clrDarkGoldenrod;
      textColor = clrYellow;
   }
   else if(lastAction == "pause")
   {
      bgColor   = clrDarkRed;
      textColor = clrRed;
   }
   else // hedge_active
   {
      bgColor   = clrDarkViolet;
      textColor = clrMagenta;
   }

   // Draw state panel in top-right corner
   string panelName = g_prefix + "STATE_BG";
   ObjectCreate(0, panelName, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, panelName, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, panelName, OBJPROP_YDISTANCE, 20);
   ObjectSetInteger(0, panelName, OBJPROP_XSIZE, 280);
   ObjectSetInteger(0, panelName, OBJPROP_YSIZE, 80);
   ObjectSetInteger(0, panelName, OBJPROP_BGCOLOR, bgColor);
   ObjectSetInteger(0, panelName, OBJPROP_BORDER_COLOR, textColor);
   ObjectSetInteger(0, panelName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);

   // Action label
   string actName = g_prefix + "STATE_ACT";
   ObjectCreate(0, actName, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, actName, OBJPROP_XDISTANCE, 270);
   ObjectSetInteger(0, actName, OBJPROP_YDISTANCE, 28);
   ObjectSetInteger(0, actName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetString(0, actName, OBJPROP_TEXT,
                   "GUARDRAIL: " + StringToUpper(lastAction));
   ObjectSetString(0, actName, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, actName, OBJPROP_FONTSIZE, 11);
   ObjectSetInteger(0, actName, OBJPROP_COLOR, textColor);

   // Regime + anomaly info
   string infoName = g_prefix + "STATE_INFO";
   ObjectCreate(0, infoName, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, infoName, OBJPROP_XDISTANCE, 270);
   ObjectSetInteger(0, infoName, OBJPROP_YDISTANCE, 50);
   ObjectSetInteger(0, infoName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetString(0, infoName, OBJPROP_TEXT,
                   "Regime: " + lastRegime + " | Anomaly: " + lastAnomaly);
   ObjectSetString(0, infoName, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, infoName, OBJPROP_FONTSIZE, 9);
   ObjectSetInteger(0, infoName, OBJPROP_COLOR, clrWhite);

   // Reason text
   string rsnName = g_prefix + "STATE_RSN";
   ObjectCreate(0, rsnName, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, rsnName, OBJPROP_XDISTANCE, 270);
   ObjectSetInteger(0, rsnName, OBJPROP_YDISTANCE, 70);
   ObjectSetInteger(0, rsnName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   // Truncate reason for display
   string displayReason = lastReason;
   if(StringLen(displayReason) > 40)
      displayReason = StringSubstr(displayReason, 0, 40) + "...";
   ObjectSetString(0, rsnName, OBJPROP_TEXT, displayReason);
   ObjectSetString(0, rsnName, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, rsnName, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, rsnName, OBJPROP_COLOR, clrSilver);
}

//+------------------------------------------------------------------+
//| Helper: skip CSV header row                                      |
//+------------------------------------------------------------------+
void SkipCSVHeader(int handle, int numCols)
{
   if(FileIsEnding(handle)) return;
   for(int i = 0; i < numCols && !FileIsLineEnding(handle) && !FileIsEnding(handle); i++)
      FileReadString(handle);
}

//+------------------------------------------------------------------+
//| Helper: parse UTC ISO-8601 string to datetime                    |
//| Handles "2024-05-14T13:30:00Z" format                           |
//+------------------------------------------------------------------+
datetime ParseUTCTime(string isoStr)
{
   // Remove trailing Z
   if(StringGetCharacter(isoStr, StringLen(isoStr) - 1) == 'Z')
      isoStr = StringSubstr(isoStr, 0, StringLen(isoStr) - 1);

   // Replace T with space
   StringReplace(isoStr, "T", " ");

   // Parse
   return StringToTime(isoStr);
}
//+------------------------------------------------------------------+
