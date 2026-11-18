//+------------------------------------------------------------------+
//| GuardrailBridgeEA.mq5                                            |
//| Time Series AI for Real-Time Trading Risk Management             |
//| Student: Mohammad Naqi Zaidi | B00963745                         |
//|                                                                  |
//| Purpose:                                                         |
//|   Reads control commands from commands/control.csv written by    |
//|   the Python guardrail controller. Executes Allow/Throttle/      |
//|   Pause/Hedge actions on the MT5 demo account. Writes back       |
//|   acknowledgements to acks/ack.csv with broker timestamps.       |
//|                                                                  |
//| File contract:                                                   |
//|   IN:  MQL5/Files/commands/control.csv                           |
//|   OUT: MQL5/Files/acks/ack.csv                                   |
//|                                                                  |
//| All controller timestamps are UTC ISO-8601.                      |
//| Only this EA converts UTC → broker time for order stamping.      |
//+------------------------------------------------------------------+
#property copyright "Mohammad Naqi Zaidi"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input string   InpSymbol          = "EURUSD";          // Symbol
input double   InpBaselineSize    = 1.0;               // Baseline lot size
input int      InpPollIntervalMs  = 1000;              // Poll interval (ms)
input int      InpAckTimeoutSec   = 2;                 // Ack timeout (seconds)
input string   InpCommandFile     = "commands\\control.csv"; // Command file path
input string   InpAckFile         = "acks\\ack.csv";         // Ack file path
input int      InpMagicNumber     = 202412;            // Magic number for orders
input int      InpSlippage        = 10;                // Max slippage (points)

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
CTrade         trade;
string         g_lastNonce       = "";       // Last processed nonce
string         g_currentAction   = "allow";  // Current guardrail state
int            g_throttleBars    = 0;        // Remaining throttle bars
int            g_pauseBars       = 0;        // Remaining pause bars
int            g_hedgeTicket     = 0;        // Hedge position ticket
int            g_hedgeAge        = 0;        // Hedge age in bars
double         g_hedgeStop       = 0.0;      // Hedge stop price
double         g_throttleMult    = 0.33;     // Throttle size multiplier
datetime       g_lastBarTime     = 0;        // Track new bars
bool           g_ackFileReady    = false;    // Ack file initialised

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpSlippage);
   trade.SetTypeFilling(ORDER_FILLING_IOC);

   // Ensure ack file exists with header
   if(!FileIsExist(InpAckFile))
   {
      int handle = FileOpen(InpAckFile, FILE_WRITE|FILE_CSV|FILE_COMMON, ',');
      if(handle != INVALID_HANDLE)
      {
         FileWrite(handle, "ts_broker", "ts_utc", "action", "status_code",
                   "status_msg", "nonce");
         FileClose(handle);
         g_ackFileReady = true;
      }
      else
      {
         // Try without FILE_COMMON
         handle = FileOpen(InpAckFile, FILE_WRITE|FILE_CSV, ',');
         if(handle != INVALID_HANDLE)
         {
            FileWrite(handle, "ts_broker", "ts_utc", "action", "status_code",
                      "status_msg", "nonce");
            FileClose(handle);
            g_ackFileReady = true;
         }
         else
         {
            Print("ERROR: Cannot create ack file: ", InpAckFile);
         }
      }
   }
   else
   {
      g_ackFileReady = true;
   }

   Print("GuardrailBridgeEA initialized. Symbol=", InpSymbol,
         " Magic=", InpMagicNumber,
         " Poll=", InpPollIntervalMs, "ms");

   // Set timer for polling
   EventSetMillisecondTimer(InpPollIntervalMs);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("GuardrailBridgeEA stopped. Reason=", reason);
}

//+------------------------------------------------------------------+
//| Timer event — poll for new commands                               |
//+------------------------------------------------------------------+
void OnTimer()
{
   PollCommands();
   ManageHedge();
   ManageTimers();
}

//+------------------------------------------------------------------+
//| Tick event — also check on each tick for responsiveness           |
//+------------------------------------------------------------------+
void OnTick()
{
   // Track new bars for timer countdown
   datetime currentBar = iTime(InpSymbol, PERIOD_M1, 0);
   if(currentBar != g_lastBarTime && g_lastBarTime != 0)
   {
      OnNewBar();
   }
   g_lastBarTime = currentBar;
}

//+------------------------------------------------------------------+
//| New bar handler — decrement action timers                        |
//+------------------------------------------------------------------+
void OnNewBar()
{
   // Decrement throttle timer
   if(g_throttleBars > 0)
   {
      g_throttleBars--;
      if(g_throttleBars == 0 && g_currentAction == "throttle")
      {
         g_currentAction = "allow";
         Print("Throttle expired → Allow");
      }
   }

   // Decrement pause timer
   if(g_pauseBars > 0)
   {
      g_pauseBars--;
      if(g_pauseBars == 0 && g_currentAction == "pause")
      {
         g_currentAction = "allow";
         Print("Pause expired → Allow");
      }
   }

   // Age the hedge
   if(g_hedgeTicket > 0)
   {
      g_hedgeAge++;
   }
}

//+------------------------------------------------------------------+
//| Poll the command file for new instructions                       |
//+------------------------------------------------------------------+
void PollCommands()
{
   if(!FileIsExist(InpCommandFile))
      return;

   int handle = FileOpen(InpCommandFile, FILE_READ|FILE_CSV|FILE_SHARE_READ, ',');
   if(handle == INVALID_HANDLE)
   {
      // Try with FILE_COMMON
      handle = FileOpen(InpCommandFile, FILE_READ|FILE_CSV|FILE_SHARE_READ|FILE_COMMON, ',');
      if(handle == INVALID_HANDLE)
         return;
   }

   // Skip header
   string headerLine = "";
   if(!FileIsEnding(handle))
   {
      // Read header fields
      for(int i = 0; i < 8 && !FileIsLineEnding(handle) && !FileIsEnding(handle); i++)
         FileReadString(handle);
   }

   // Read all rows, process the last one with a new nonce
   string lastTs      = "";
   string lastSymbol  = "";
   string lastAction  = "";
   string lastParams  = "";
   string lastConfig  = "";
   string lastReason  = "";
   string lastText    = "";
   string lastNonce   = "";
   bool   foundNew    = false;

   while(!FileIsEnding(handle))
   {
      string ts     = FileReadString(handle); if(FileIsEnding(handle) && ts == "") break;
      string sym    = FileReadString(handle);
      string act    = FileReadString(handle);
      string par    = FileReadString(handle);
      string cfg    = FileReadString(handle);
      string rsn    = FileReadString(handle);
      string txt    = FileReadString(handle);
      string nonce  = FileReadString(handle);

      if(nonce != "" && nonce != g_lastNonce)
      {
         lastTs     = ts;
         lastSymbol = sym;
         lastAction = act;
         lastParams = par;
         lastConfig = cfg;
         lastReason = rsn;
         lastText   = txt;
         lastNonce  = nonce;
         foundNew   = true;
      }
   }

   FileClose(handle);

   if(foundNew && lastNonce != "")
   {
      ProcessCommand(lastTs, lastSymbol, lastAction, lastParams,
                     lastConfig, lastReason, lastText, lastNonce);
   }
}

//+------------------------------------------------------------------+
//| Process a single command                                         |
//+------------------------------------------------------------------+
void ProcessCommand(string ts_utc, string symbol, string action,
                    string params_json, string config_version,
                    string reason_code, string reason_text, string nonce)
{
   g_lastNonce = nonce;

   Print("CMD received: action=", action, " nonce=", nonce,
         " reason=", reason_code);

   int statusCode = 0;
   string statusMsg = "ok";

   // ── Execute the action ──────────────────────────────────────────
   if(action == "allow")
   {
      ExecuteAllow();
   }
   else if(action == "throttle")
   {
      ExecuteThrottle(params_json);
   }
   else if(action == "pause")
   {
      ExecutePause();
   }
   else if(action == "hedge_active")
   {
      statusCode = ExecuteHedge(params_json);
      if(statusCode != 0)
         statusMsg = "hedge_failed";
   }
   else if(action == "close_hedge")
   {
      CloseHedge("command");
   }
   else
   {
      Print("Unknown action: ", action);
      statusCode = -1;
      statusMsg = "unknown_action";
   }

   // ── Write acknowledgement ───────────────────────────────────────
   WriteAck(ts_utc, action, statusCode, statusMsg, nonce);
}

//+------------------------------------------------------------------+
//| Execute: Allow — clear all restrictions                          |
//+------------------------------------------------------------------+
void ExecuteAllow()
{
   g_currentAction = "allow";
   g_throttleBars  = 0;
   g_pauseBars     = 0;
   Print("Action: ALLOW — all restrictions cleared");
}

//+------------------------------------------------------------------+
//| Execute: Throttle — reduce position size                         |
//+------------------------------------------------------------------+
void ExecuteThrottle(string params_json)
{
   g_currentAction = "throttle";
   g_throttleBars  = 5;  // default from config

   // Parse size_mult from params if available
   double sizeMult = ExtractDouble(params_json, "size_mult", 0.33);
   g_throttleMult  = sizeMult;

   Print("Action: THROTTLE — size_mult=", DoubleToString(sizeMult, 2),
         " for ", g_throttleBars, " bars");
}

//+------------------------------------------------------------------+
//| Execute: Pause — block new entries                               |
//+------------------------------------------------------------------+
void ExecutePause()
{
   g_currentAction = "pause";
   g_pauseBars     = 10;  // default from config

   Print("Action: PAUSE — new entries blocked for ", g_pauseBars, " bars");
}

//+------------------------------------------------------------------+
//| Execute: Hedge — open a small counter position                   |
//+------------------------------------------------------------------+
int ExecuteHedge(string params_json)
{
   string direction = ExtractString(params_json, "direction", "long");
   double sizeMult  = ExtractDouble(params_json, "size_mult", 0.10);
   double stopPrice = ExtractDouble(params_json, "stop_price", 0.0);

   double lotSize = NormalizeDouble(InpBaselineSize * sizeMult, 2);
   if(lotSize < SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MIN))
      lotSize = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MIN);

   double price;
   ENUM_ORDER_TYPE orderType;

   if(direction == "long")
   {
      price     = SymbolInfoDouble(InpSymbol, SYMBOL_ASK);
      orderType = ORDER_TYPE_BUY;
   }
   else
   {
      price     = SymbolInfoDouble(InpSymbol, SYMBOL_BID);
      orderType = ORDER_TYPE_SELL;
   }

   // Normalize stop price
   int digits = (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS);
   stopPrice  = NormalizeDouble(stopPrice, digits);
   price      = NormalizeDouble(price, digits);

   bool result;
   if(stopPrice > 0)
      result = trade.PositionOpen(InpSymbol, orderType, lotSize, price,
                                  stopPrice, 0.0, "Guardrail hedge");
   else
      result = trade.PositionOpen(InpSymbol, orderType, lotSize, price,
                                  0.0, 0.0, "Guardrail hedge");

   if(result)
   {
      g_hedgeTicket = (int)trade.ResultOrder();
      g_hedgeAge    = 0;
      g_hedgeStop   = stopPrice;
      g_currentAction = "hedge_active";

      Print("HEDGE opened: ticket=", g_hedgeTicket,
            " dir=", direction, " lots=", DoubleToString(lotSize, 2),
            " stop=", DoubleToString(stopPrice, digits));
      return 0;
   }
   else
   {
      Print("HEDGE FAILED: error=", GetLastError(),
            " retcode=", trade.ResultRetcode());
      return (int)trade.ResultRetcode();
   }
}

//+------------------------------------------------------------------+
//| Close the active hedge position                                  |
//+------------------------------------------------------------------+
void CloseHedge(string reason)
{
   if(g_hedgeTicket <= 0) return;

   if(trade.PositionClose(g_hedgeTicket))
   {
      Print("HEDGE closed: ticket=", g_hedgeTicket,
            " age=", g_hedgeAge, " reason=", reason);
   }
   else
   {
      Print("HEDGE close failed: ticket=", g_hedgeTicket,
            " error=", GetLastError());
   }

   g_hedgeTicket   = 0;
   g_hedgeAge      = 0;
   g_currentAction = "allow";
}

//+------------------------------------------------------------------+
//| Manage hedge: check age and stop                                 |
//+------------------------------------------------------------------+
void ManageHedge()
{
   if(g_hedgeTicket <= 0) return;

   // Check max age (10 bars)
   if(g_hedgeAge >= 10)
   {
      CloseHedge("max_age");
      return;
   }

   // Position might have been closed by stop
   if(!PositionSelectByTicket(g_hedgeTicket))
   {
      Print("Hedge position ", g_hedgeTicket, " no longer exists (stop hit or manual close)");
      g_hedgeTicket   = 0;
      g_hedgeAge      = 0;
      g_currentAction = "allow";
   }
}

//+------------------------------------------------------------------+
//| Manage throttle/pause timers on each poll                        |
//+------------------------------------------------------------------+
void ManageTimers()
{
   // Timers are decremented in OnNewBar()
   // This function handles any additional timer logic
}

//+------------------------------------------------------------------+
//| Write acknowledgement to ack file                                |
//+------------------------------------------------------------------+
void WriteAck(string ts_utc, string action, int statusCode,
              string statusMsg, string nonce)
{
   int handle = FileOpen(InpAckFile, FILE_READ|FILE_WRITE|FILE_CSV|FILE_SHARE_READ, ',');
   if(handle == INVALID_HANDLE)
   {
      // Try with FILE_COMMON
      handle = FileOpen(InpAckFile, FILE_READ|FILE_WRITE|FILE_CSV|FILE_SHARE_READ|FILE_COMMON, ',');
   }

   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open ack file for writing: ", InpAckFile);
      return;
   }

   // Seek to end
   FileSeek(handle, 0, SEEK_END);

   // Broker timestamp
   string tsBroker = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);

   FileWrite(handle, tsBroker, ts_utc, action,
             IntegerToString(statusCode), statusMsg, nonce);

   FileClose(handle);

   Print("ACK written: nonce=", nonce, " status=", statusCode, " (", statusMsg, ")");
}

//+------------------------------------------------------------------+
//| Utility: check if new entries are allowed                        |
//+------------------------------------------------------------------+
bool IsEntryAllowed()
{
   if(g_currentAction == "pause")
      return false;
   return true;
}

//+------------------------------------------------------------------+
//| Utility: get current position size multiplier                    |
//+------------------------------------------------------------------+
double GetSizeMultiplier()
{
   if(g_currentAction == "throttle")
      return g_throttleMult;
   return 1.0;
}

//+------------------------------------------------------------------+
//| Utility: extract a double from a simple JSON-like string         |
//| e.g. {"size_mult": 0.33, "stop_price": 1.09500}                 |
//+------------------------------------------------------------------+
double ExtractDouble(string json, string key, double defaultVal)
{
   string searchKey = "\"" + key + "\"";
   int pos = StringFind(json, searchKey);
   if(pos < 0) return defaultVal;

   // Find the colon after the key
   int colonPos = StringFind(json, ":", pos + StringLen(searchKey));
   if(colonPos < 0) return defaultVal;

   // Find the value start (skip spaces)
   int valStart = colonPos + 1;
   while(valStart < StringLen(json) &&
         (StringGetCharacter(json, valStart) == ' ' ||
          StringGetCharacter(json, valStart) == '"'))
      valStart++;

   // Find the value end
   int valEnd = valStart;
   while(valEnd < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, valEnd);
      if(ch == ',' || ch == '}' || ch == '"' || ch == ' ')
         break;
      valEnd++;
   }

   string valStr = StringSubstr(json, valStart, valEnd - valStart);
   return StringToDouble(valStr);
}

//+------------------------------------------------------------------+
//| Utility: extract a string value from simple JSON                 |
//+------------------------------------------------------------------+
string ExtractString(string json, string key, string defaultVal)
{
   string searchKey = "\"" + key + "\"";
   int pos = StringFind(json, searchKey);
   if(pos < 0) return defaultVal;

   int colonPos = StringFind(json, ":", pos + StringLen(searchKey));
   if(colonPos < 0) return defaultVal;

   // Find opening quote of value
   int quoteStart = StringFind(json, "\"", colonPos + 1);
   if(quoteStart < 0) return defaultVal;

   // Find closing quote
   int quoteEnd = StringFind(json, "\"", quoteStart + 1);
   if(quoteEnd < 0) return defaultVal;

   return StringSubstr(json, quoteStart + 1, quoteEnd - quoteStart - 1);
}

//+------------------------------------------------------------------+
//| Info panel on chart                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam,
                  const double& dparam, const string& sparam)
{
   // Optional: update chart panel with current guardrail state
}
//+------------------------------------------------------------------+
