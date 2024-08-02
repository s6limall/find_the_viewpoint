
"use strict";

let IsInRemoteControl = require('./IsInRemoteControl.js')
let RawRequest = require('./RawRequest.js')
let GetProgramState = require('./GetProgramState.js')
let IsProgramRunning = require('./IsProgramRunning.js')
let AddToLog = require('./AddToLog.js')
let GetSafetyMode = require('./GetSafetyMode.js')
let Load = require('./Load.js')
let IsProgramSaved = require('./IsProgramSaved.js')
let Popup = require('./Popup.js')
let GetRobotMode = require('./GetRobotMode.js')
let GetLoadedProgram = require('./GetLoadedProgram.js')

module.exports = {
  IsInRemoteControl: IsInRemoteControl,
  RawRequest: RawRequest,
  GetProgramState: GetProgramState,
  IsProgramRunning: IsProgramRunning,
  AddToLog: AddToLog,
  GetSafetyMode: GetSafetyMode,
  Load: Load,
  IsProgramSaved: IsProgramSaved,
  Popup: Popup,
  GetRobotMode: GetRobotMode,
  GetLoadedProgram: GetLoadedProgram,
};
