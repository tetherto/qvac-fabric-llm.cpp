'use strict'

const path = require('path')

const COMPARISON_DIR = __dirname ? path.resolve(__dirname, '..') : path.resolve(__dirname)
const LIB_DIR = path.join(COMPARISON_DIR, 'lib')

function comparisonRoot () {
  return path.resolve(__dirname, '..')
}

function engineDir () {
  return path.resolve(comparisonRoot(), '../..')
}

function repositoryDir () {
  return path.resolve(engineDir(), '../..')
}

function workspaceDir () {
  return path.resolve(repositoryDir(), '..')
}

module.exports = {
  comparisonRoot,
  engineDir,
  repositoryDir,
  workspaceDir,
  LIB_DIR
}
