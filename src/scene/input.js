/**
 * Pointer input for the arcball.
 *
 * Mouse and touch rather than pointer events: some embedders only synthesise the
 * former, and being unreachable by input is a worse failure than the small
 * duplication here. Move/up live on `window` so a drag survives leaving the
 * canvas.
 *
 * Drag deltas ACCUMULATE into a virtual cursor rather than tracking the pointer
 * absolutely, which is what makes it feel like an arcball instead of a slider.
 */

/** Keys the page consumes, so the rest reach the browser untouched. */
const FLIGHT_KEYS = new Set([
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ',
  'w', 'a', 's', 'd', 'q', 'e',
]);
// 'c' is handled but not listed: it toggles the camera and is not a flight input.

export class Input {
  constructor(canvas) {
    this.x = 0;
    this.y = 0;
    this.down = false;
    this.everUsed = false;
    this._px = 0;
    this._py = 0;
    // Reused: this is read exactly once per frame, forever.
    this._cmd = { pitch: 0, yaw: 0, roll: 0, thrust: 0, fire: false, trigger: false };
    this._state = {
      dragging: false, everUsed: false, x: 0, y: 0, width: 0, height: 0,
      // The flight command rides along on the frame state, because that object is
      // what actually crosses into the renderer — `state()` is the boundary.
      cmd: this._cmd, chase: false,
    };

    canvas.style.touchAction = 'none';
    canvas.style.cursor = 'grab';
    this.canvas = canvas;

    const start = (cx, cy) => {
      this.down = true;
      this._px = cx;
      this._py = cy;
      if (!this.everUsed) {
        // Begin from the centre so the first drag is a small nudge, not a jump.
        this.x = canvas.width * 0.5;
        this.y = canvas.height * 0.5;
        this.everUsed = true;
      }
      canvas.style.cursor = 'grabbing';
    };
    const move = (cx, cy) => {
      if (!this.down) return;
      const s = canvas.width / Math.max(1, canvas.clientWidth);
      this.x += (cx - this._px) * s;
      this.y -= (cy - this._py) * s;          // screen y is down, ours is up
      this.y = Math.max(1, Math.min(canvas.height - 1, this.y));   // clamp pitch
      this._px = cx;
      this._py = cy;
    };
    const end = () => {
      this.down = false;
      canvas.style.cursor = 'grab';
    };

    // Keyboard, for flying. Held state rather than events, because the physics wants
    // to know what is being commanded on every step, not when it changed.
    this.keys = new Set();
    /** Latches on the first flight key, so the camera knows to start chasing. */
    this.everFlown = false;
    /** Whether the camera is chasing the ship. Latches on the first flight key; C toggles. */
    this.chase = false;
    this._cHeld = false;
    this._spaceHeld = false;
    this._fireEdge = false;
    const key = (e, down) => {
      // Never swallow a browser shortcut.
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (down) {
        this.keys.add(k);
        if (FLIGHT_KEYS.has(k)) { this.everFlown = true; this.chase = true; }
        // C toggles between chasing the ship and the free orbit camera. On keydown
        // only, so holding it does not strobe.
        if (k === 'c' && !this._cHeld) { this.chase = !this.chase; this._cHeld = true; }
        // Space is a TRIGGER now, not a button: holding it charges and releasing it fires. The edge
        // is still latched, but only so a tap cannot be missed by a frame that lands between the
        // keydown and the keyup — the shot itself leaves on release, where the charge is known.
        if (k === ' ' && !this._spaceHeld) { this._fireEdge = true; this._spaceHeld = true; }
      } else {
        this.keys.delete(k);
        if (k === 'c') this._cHeld = false;
        if (k === ' ') this._spaceHeld = false;
      }
      if (FLIGHT_KEYS.has(k)) e.preventDefault();
    };
    window.addEventListener('keydown', (e) => key(e, true));
    window.addEventListener('keyup', (e) => key(e, false));
    // Losing focus mid-manoeuvre would otherwise leave a key stuck down forever.
    window.addEventListener('blur', () => this.keys.clear());

    canvas.addEventListener('mousedown', (e) => { start(e.clientX, e.clientY); e.preventDefault(); });
    window.addEventListener('mousemove', (e) => move(e.clientX, e.clientY));
    window.addEventListener('mouseup', end);
    canvas.addEventListener('touchstart', (e) => {
      const t = e.touches[0];
      if (t) start(t.clientX, t.clientY);
      e.preventDefault();
    }, { passive: false });
    window.addEventListener('touchmove', (e) => {
      const t = e.touches[0];
      if (t) move(t.clientX, t.clientY);
    }, { passive: false });
    window.addEventListener('touchend', end);
  }

  /**
   * Flight command, each axis in -1..1. Reused, like `state()`.
   * Arrows or WASD pitch and yaw; Q/E roll; space or W thrusts.
   */
  command() {
    const k = this.keys;
    const ax = (neg, pos) => (k.has(neg) ? -1 : 0) + (k.has(pos) ? 1 : 0);
    const c = this._cmd;
    c.pitch = ax('ArrowDown', 'ArrowUp') + ax('s', 'w');
    c.yaw = ax('ArrowLeft', 'ArrowRight') + ax('a', 'd');
    c.roll = ax('e', 'q');
    // Thrust is up/W only now: space is the trigger. `pitch` already carries the
    // accelerate axis, so this stays at zero rather than double-counting it.
    c.thrust = 0;
    // Consumed on read, so one press is one event however many frames the key is down for. The
    // railgun turns these two into shots: `trigger` integrates the charge, `fire` only guarantees a
    // press shorter than a frame is still seen.
    c.fire = this._fireEdge;
    this._fireEdge = false;
    c.trigger = this._spaceHeld;
    c.pitch = Math.max(-1, Math.min(1, c.pitch));
    c.yaw = Math.max(-1, Math.min(1, c.yaw));
    return c;
  }

  /** Pointer state. The object is REUSED — read it, do not retain it. */
  state(canvas) {
    const s = this._state;
    s.dragging = this.down;
    s.everUsed = this.everUsed;
    s.x = this.x;
    s.y = this.y;
    s.width = canvas.width;
    s.height = canvas.height;
    s.chase = this.chase;
    this.command();                 // refreshes s.cmd in place
    return s;
  }
}
