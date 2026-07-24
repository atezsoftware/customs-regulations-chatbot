import assert from 'node:assert/strict';
import test from 'node:test';
import {parseEffort} from './chat-messages.controller';

test('parseEffort defaults to low when omitted', () => {
  // "low" matches the pre-effort baseline exactly, so any caller that omits
  // `effort` (an older client, a dropped query param) never regresses.
  assert.equal(parseEffort(undefined), 'low');
});

test('parseEffort passes through valid effort levels', () => {
  assert.equal(parseEffort('low'), 'low');
  assert.equal(parseEffort('medium'), 'medium');
  assert.equal(parseEffort('high'), 'high');
});

test('parseEffort rejects an unknown effort level', () => {
  assert.throws(() => parseEffort('extreme'), /Invalid effort level/);
});
