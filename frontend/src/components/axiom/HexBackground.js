import React, { useEffect, useRef } from 'react';

export const HexBackground = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let animationId;
    let hexCenters = [];
    let activeZone = null;
    let zoneIntensity = 0;
    let hallucinationPulse = null;
    let cacheHitPulse = null;

    const hexRadius = 32;
    const hexGap = 2;

    const colors = {
      retrieval: 'rgb(217, 119, 6)',    // amber
      generation: 'rgb(124, 58, 237)',   // violet
      evaluation: 'rgb(5, 150, 105)',    // emerald
      error: 'rgb(220, 38, 38)',         // red
    };

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      calculateHexCenters();
    };

    const calculateHexCenters = () => {
      hexCenters = [];
      const w = canvas.width;
      const h = canvas.height;
      
      const hexWidth = Math.sqrt(3) * (hexRadius + hexGap);
      const hexHeight = 2 * (hexRadius + hexGap);
      const vertOffset = hexHeight * 0.75;

      for (let row = 0; row * vertOffset < h + hexRadius; row++) {
        const offsetX = row % 2 === 0 ? 0 : hexWidth / 2;
        for (let col = 0; col * hexWidth < w + hexRadius; col++) {
          const x = col * hexWidth + offsetX;
          const y = row * vertOffset;
          
          let zone = 'center';
          if (x / w < 0.3) zone = 'left';
          else if (x / w > 0.7) zone = 'right';
          
          hexCenters.push({ x, y, zone, intensity: 0 });
        }
      }
    };

    const drawHex = (cx, cy, radius, color, intensity) => {
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i - Math.PI / 6;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 0.8 + intensity * 0.5;
      ctx.stroke();
    };

    const toRGBA = (rgbStr, alpha) => {
      return rgbStr.replace('rgb(', 'rgba(').replace(')', `, ${alpha})`);
    };

    const render = (time) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const breathe = 0.03 + Math.sin(time / 2000) * 0.005;

      hexCenters.forEach((hex) => {
        let baseColor = colors.generation;
        let intensity = breathe;

        // Zone-based default coloring
        if (hex.zone === 'left') baseColor = colors.retrieval;
        else if (hex.zone === 'center') baseColor = colors.generation;
        else if (hex.zone === 'right') baseColor = colors.evaluation;

        // Active zone highlighting
        if (activeZone) {
          const zoneMatch = 
            (activeZone === 'retrieval' && hex.zone === 'left') ||
            (activeZone === 'generation' && hex.zone === 'center') ||
            (activeZone === 'evaluation' && hex.zone === 'right');

          if (zoneMatch) {
            intensity = Math.min(0.4, zoneIntensity);
          }
        }

        // Hallucination pulse effect
        if (hallucinationPulse) {
          const dist = Math.sqrt(
            Math.pow(hex.x - canvas.width / 2, 2) + 
            Math.pow(hex.y - canvas.height / 2, 2)
          );
          const pulseRadius = hallucinationPulse.radius;
          if (dist < pulseRadius && dist > pulseRadius - 100) {
            baseColor = colors.error;
            intensity = 0.5 * (1 - (pulseRadius - dist) / 100);
          }
        }

        // Cache hit pulse: violet flash on left (retrieval) zone
        if (cacheHitPulse && hex.zone === 'left') {
          const elapsed = time - cacheHitPulse.startTime;
          if (elapsed < 1500) {
            const progress = elapsed / 1500;
            const pulseIntensity = 0.6 * Math.sin(progress * Math.PI);
            if (pulseIntensity > intensity) {
              baseColor = colors.generation; // violet
              intensity = pulseIntensity;
            }
          } else {
            cacheHitPulse = null;
          }
        }

        drawHex(hex.x, hex.y, hexRadius, toRGBA(baseColor, intensity), intensity);
      });

      // Update hallucination pulse
      if (hallucinationPulse) {
        hallucinationPulse.radius += 8;
        if (hallucinationPulse.radius > Math.max(canvas.width, canvas.height)) {
          hallucinationPulse = null;
        }
      }

      animationId = requestAnimationFrame(render);
    };

    window.axiomSetZone = (zone, intensity) => {
      activeZone = zone;
      zoneIntensity = intensity;
    };

    window.axiomHallucinationPulse = () => {
      hallucinationPulse = { radius: 0 };
    };

    window.axiomCacheHit = () => {
      cacheHitPulse = { startTime: performance.now() };
    };

    resize();
    window.addEventListener('resize', resize);
    animationId = requestAnimationFrame(render);

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationId);
      delete window.axiomSetZone;
      delete window.axiomHallucinationPulse;
      delete window.axiomCacheHit;
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      id="axiom-hex-bg"
      data-testid="hex-background"
    />
  );
};

export default HexBackground;
