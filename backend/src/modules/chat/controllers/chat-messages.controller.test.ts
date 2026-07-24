import assert from 'node:assert/strict';
import test from 'node:test';
import {parseEffort} from './chat-messages.controller';

test('parseEffort defaults to medium when omitted', () => {
  assert.equal(parseEffort(undefined), 'medium');
});

test('parseEffort passes through valid effort levels', () => {
  assert.equal(parseEffort('low'), 'low');
  assert.equal(parseEffort('medium'), 'medium');
  assert.equal(parseEffort('high'), 'high');
});

test('parseEffort rejects an unknown effort level', () => {
  assert.throws(() => parseEffort('extreme'), /Invalid effort level/);
});
