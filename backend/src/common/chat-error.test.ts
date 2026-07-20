import assert from 'node:assert/strict';
import test from 'node:test';
import {formatChatError} from './chat-error';

test('formats an Error with its diagnostic message', () => {
  assert.equal(
    formatChatError(new Error('Core stream closed before completion.')),
    'Core stream closed before completion.',
  );
});

test('formats unknown and empty failures without producing an empty admin error', () => {
  assert.equal(formatChatError('provider returned HTTP 503'), 'provider returned HTTP 503');
  assert.equal(formatChatError(new Error('')), 'Unknown chat error.');
});

test('removes NUL bytes and bounds unusually large provider errors', () => {
  const formatted = formatChatError(
    new Error(`prefix${String.fromCharCode(0)}${'x'.repeat(3000)}`),
  );
  assert.equal(formatted.length, 2000);
  assert.equal(formatted.includes(String.fromCharCode(0)), false);
});
