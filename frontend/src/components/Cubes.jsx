import { useCallback, useEffect, useRef } from 'react';
import gsap from 'gsap';
import './Cubes.css';

const Cubes = ({
  gridSize      = 10,
  cubeSize,
  maxAngle      = 45,
  radius        = 3,
  easing        = 'power3.out',
  duration      = { enter: 0.3, leave: 0.6 },
  cellGap,
  borderStyle   = '1px solid rgba(1,51,40,0.55)',
  faceColor     = '#013328',
  shadow        = false,
  autoAnimate   = true,
  rippleOnClick = true,
  rippleColor   = '#CC8B65',
  rippleSpeed   = 2,
  imageUrl,
}) => {
  const sceneRef      = useRef(null);
  const rafRef        = useRef(null);
  const idleTimerRef  = useRef(null);
  const userActiveRef = useRef(false);
  const simPosRef     = useRef({ x: 0, y: 0 });
  const simTargetRef  = useRef({ x: 0, y: 0 });
  const simRAFRef     = useRef(null);

  const colGap =
    typeof cellGap === 'number'
      ? `${cellGap}px`
      : cellGap?.col !== undefined
      ? `${cellGap.col}px`
      : '3%';
  const rowGap =
    typeof cellGap === 'number'
      ? `${cellGap}px`
      : cellGap?.row !== undefined
      ? `${cellGap.row}px`
      : '3%';

  const enterDur = duration.enter;
  const leaveDur = duration.leave;

  const tiltAt = useCallback(
    (rowCenter, colCenter) => {
      if (!sceneRef.current) return;
      sceneRef.current.querySelectorAll('.cube').forEach(cube => {
        const r    = +cube.dataset.row;
        const c    = +cube.dataset.col;
        const dist = Math.hypot(r - rowCenter, c - colCenter);
        if (dist <= radius) {
          const pct   = 1 - dist / radius;
          const angle = pct * maxAngle;
          gsap.to(cube, { duration: enterDur, ease: easing, overwrite: true, rotateX: -angle, rotateY: angle });
        } else {
          gsap.to(cube, { duration: leaveDur, ease: 'power3.out', overwrite: true, rotateX: 0, rotateY: 0 });
        }
      });
    },
    [radius, maxAngle, enterDur, leaveDur, easing],
  );

  const onPointerMove = useCallback(
    e => {
      userActiveRef.current = true;
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
      const rect      = sceneRef.current.getBoundingClientRect();
      const cellW     = rect.width  / gridSize;
      const cellH     = rect.height / gridSize;
      const colCenter = (e.clientX - rect.left) / cellW;
      const rowCenter = (e.clientY - rect.top)  / cellH;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => tiltAt(rowCenter, colCenter));
      idleTimerRef.current = setTimeout(() => { userActiveRef.current = false; }, 3000);
    },
    [gridSize, tiltAt],
  );

  const resetAll = useCallback(() => {
    if (!sceneRef.current) return;
    sceneRef.current.querySelectorAll('.cube').forEach(cube =>
      gsap.to(cube, { duration: leaveDur, rotateX: 0, rotateY: 0, ease: 'power3.out' }),
    );
  }, [leaveDur]);

  const onTouchMove = useCallback(
    e => {
      e.preventDefault();
      userActiveRef.current = true;
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
      const rect      = sceneRef.current.getBoundingClientRect();
      const touch     = e.touches[0];
      const colCenter = (touch.clientX - rect.left) / (rect.width  / gridSize);
      const rowCenter = (touch.clientY - rect.top)  / (rect.height / gridSize);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => tiltAt(rowCenter, colCenter));
      idleTimerRef.current = setTimeout(() => { userActiveRef.current = false; }, 3000);
    },
    [gridSize, tiltAt],
  );

  const onTouchEnd = useCallback(() => { resetAll(); }, [resetAll]);

  const onClick = useCallback(
    e => {
      if (!rippleOnClick || !sceneRef.current) return;
      const rect   = sceneRef.current.getBoundingClientRect();
      const cellW  = rect.width  / gridSize;
      const cellH  = rect.height / gridSize;
      const colHit = Math.floor((e.clientX - rect.left) / cellW);
      const rowHit = Math.floor((e.clientY - rect.top)  / cellH);

      const spreadDelay  = (0.15 / rippleSpeed);
      const animDuration = (0.3  / rippleSpeed);
      const holdTime     = (0.6  / rippleSpeed);

      const rings = {};
      sceneRef.current.querySelectorAll('.cube').forEach(cube => {
        const dist = Math.hypot(+cube.dataset.row - rowHit, +cube.dataset.col - colHit);
        const ring = Math.round(dist);
        if (!rings[ring]) rings[ring] = [];
        rings[ring].push(cube);
      });

      Object.keys(rings).map(Number).sort((a, b) => a - b).forEach(ring => {
        const delay = ring * spreadDelay;
        const faces = rings[ring].flatMap(c => Array.from(c.querySelectorAll('.cube-face')));
        gsap.to(faces, { backgroundColor: rippleColor, duration: animDuration, delay, ease: 'power3.out' });
        gsap.to(faces, { backgroundColor: faceColor,   duration: animDuration, delay: delay + animDuration + holdTime, ease: 'power3.out' });
      });
    },
    [rippleOnClick, gridSize, faceColor, rippleColor, rippleSpeed],
  );

  useEffect(() => {
    if (!autoAnimate || !sceneRef.current) return;
    simPosRef.current    = { x: Math.random() * gridSize, y: Math.random() * gridSize };
    simTargetRef.current = { x: Math.random() * gridSize, y: Math.random() * gridSize };
    const speed = 0.018;
    const loop  = () => {
      if (!userActiveRef.current) {
        const pos = simPosRef.current;
        const tgt = simTargetRef.current;
        pos.x += (tgt.x - pos.x) * speed;
        pos.y += (tgt.y - pos.y) * speed;
        tiltAt(pos.y, pos.x);
        if (Math.hypot(pos.x - tgt.x, pos.y - tgt.y) < 0.1) {
          simTargetRef.current = { x: Math.random() * gridSize, y: Math.random() * gridSize };
        }
      }
      simRAFRef.current = requestAnimationFrame(loop);
    };
    simRAFRef.current = requestAnimationFrame(loop);
    return () => { if (simRAFRef.current != null) cancelAnimationFrame(simRAFRef.current); };
  }, [autoAnimate, gridSize, tiltAt]);

  useEffect(() => {
    const el = sceneRef.current;
    if (!el) return;
    el.addEventListener('pointermove', onPointerMove);
    el.addEventListener('pointerleave', resetAll);
    el.addEventListener('click', onClick);
    el.addEventListener('touchmove',  onTouchMove,  { passive: false });
    el.addEventListener('touchstart', () => { userActiveRef.current = true; }, { passive: true });
    el.addEventListener('touchend',   onTouchEnd,   { passive: true });
    return () => {
      el.removeEventListener('pointermove',  onPointerMove);
      el.removeEventListener('pointerleave', resetAll);
      el.removeEventListener('click',        onClick);
      el.removeEventListener('touchmove',    onTouchMove);
      el.removeEventListener('touchend',     onTouchEnd);
      if (rafRef.current)      cancelAnimationFrame(rafRef.current);
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    };
  }, [onPointerMove, resetAll, onClick, onTouchMove, onTouchEnd]);

  const cells = Array.from({ length: gridSize });

  const sceneStyle = {
    gridTemplateColumns: cubeSize ? `repeat(${gridSize}, ${cubeSize}px)` : `repeat(${gridSize}, 1fr)`,
    gridTemplateRows:    cubeSize ? `repeat(${gridSize}, ${cubeSize}px)` : `repeat(${gridSize}, 1fr)`,
    columnGap: colGap,
    rowGap:    rowGap,
  };

  const wrapperStyle = {
    '--cube-face-border': borderStyle,
    '--cube-face-bg':     faceColor,
    '--cube-face-shadow': shadow === true ? '0 0 6px rgba(0,0,0,.5)' : shadow || 'none',
    width:  '100%',
    height: '100%',
    ...(cubeSize
      ? { width: `${gridSize * cubeSize}px`, height: `${gridSize * cubeSize}px` }
      : {}),
  };

  return (
    <div className="cubes-wrapper" style={wrapperStyle}>
      <div ref={sceneRef} className="cubes-scene" style={sceneStyle}>
        {cells.map((_, r) =>
          cells.map((__, c) => {
            const frontStyle = imageUrl
              ? {
                  backgroundImage:    `url(${imageUrl})`,
                  backgroundSize:     `${gridSize * 100}% ${gridSize * 100}%`,
                  backgroundPosition: `${(c / Math.max(gridSize - 1, 1)) * 100}% ${(r / Math.max(gridSize - 1, 1)) * 100}%`,
                  backgroundColor:    'transparent',
                }
              : {};

            return (
              <div key={`${r}-${c}`} className="cube" data-row={r} data-col={c}>
                <div className="cube-face cube-face--top" />
                <div className="cube-face cube-face--bottom" />
                <div className="cube-face cube-face--left" />
                <div className="cube-face cube-face--right" />
                <div className="cube-face cube-face--front" style={frontStyle} />
                <div className="cube-face cube-face--back" />
              </div>
            );
          }),
        )}
      </div>
    </div>
  );
};

export default Cubes;
