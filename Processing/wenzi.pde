// Info Smog — 信息雾霾文字粒子版（优化版）
// 优化要点:
// 1. 减少排序频率 (每 2 帧排一次)
// 2. PGraphics 离屏缓存常用文字
// 3. 降低粒子密度
// 4. 简化物理计算
// 5. 空格键模拟 depth camera 靠近效果

ArrayList<TextParticle> particles;
PFont font;
int MAX_PARTICLES;
float globalPulse = 0;

float noiseOffsetX = 0.0;
float noiseOffsetY = 100.0;

// 文字缓存
HashMap<String, PGraphics> textCache;

// Depth Camera 模拟
float depthProximity = 0.0; // 0.0 = 远离，1.0 = 靠近
float targetProximity = 0.0;

// 可调参数 (独立按键控制)
float flowSpeedMultiplier = 1.0;  // 流动速度倍数 (X加速)
float targetFlowSpeed = 1.0;
float jitterMultiplier = 1.0;      // 抖动强度倍数 (Z加速)
float targetJitter = 1.0;

String[] slogans = {
  "SALE", "LIMITED TIME", "Breaking News", "MUST WATCH",
  "RECOMMENDED FOR YOU", "Hot Now", "New Post", "Live",
  "Only Today", "DON'T MISS", "Flash Deal", "Sponsored",
  "UP NEXT", "Trending", "Click Here", "Subscribe",
  "特价", "爆款", "猜你喜欢", "今日热搜",
  "立即购买", "限时优惠", "新消息", "你的好友刚刚点赞",
  "広告", "おすすめ", "今だけ", "新着",
  "SCROLL", "MORE", "Tap to Open", "For You",
  "BREAKING", "UPDATE", "Watch Now", "Click to Learn More",
  "FREE SHIPPING", "24H ONLY", "Last Chance", "Act Now",
  "New Arrival", "Best Seller", "Top Rated", "You Missed",
  "Exclusive", "Premium", "Limited Stock", "Sold Out Soon",
  "加入购物车", "秒杀", "拼单", "直播中",
  "关注", "收藏", "转发", "评论",
  "会员专享", "满减", "包邮", "预售",
  "涨粉", "热榜", "置顶", "精选",
  "お得", "期間限定", "人気",
  "送料無料", 
  "SWIPE UP", "Tag Someone", "Double Tap", "Share Now",
  "Join Now", "Sign Up", "Download", "Install",
  "Turn On Notifications", "Follow", "Like", "Comment",
  "Add to Cart", "Buy Now", "Pre-order", "Reserve",
  "Clearance", "Mega Sale", "Today Only", "Hurry",
  "Deal of the Day", "Staff Pick", "Editor's Choice", "Verified"
};

void setup() {
  fullScreen(P2D);
  smooth(4);
  
  // 尝试加载支持中日文的系统字体
  String[] fontOptions = {
    "Microsoft YaHei",    // Windows 中文
    "SimHei",             // Windows 中文
    "PingFang SC",        // macOS 中文
    "Noto Sans CJK SC",   // Linux 中文
    "Arial Unicode MS",   // 通用
    "Sans Serif"          // 后备
  };
  
  font = null;
  for (String fontName : fontOptions) {
    try {
      font = createFont(fontName, 24, true);
      if (font != null) {
        println("Using font: " + fontName);
        break;
      }
    } catch (Exception e) {
      continue;
    }
  }
  
  if (font == null) {
    font = createFont("Sans Serif", 24, true);
  }
  
  textFont(font);
  
  particles = new ArrayList<TextParticle>();
  textCache = new HashMap<String, PGraphics>();
  
  // 适度提高密度
  float density = 0.00016;
  MAX_PARTICLES = int(width * height * density);
  MAX_PARTICLES = min(MAX_PARTICLES, 380);
  
  // 预渲染文字到缓存
  for (String s : slogans) {
    cacheText(s);
  }
  
  // 初始化粒子
  for (int i = 0; i < MAX_PARTICLES; i++) {
    String s = slogans[i % slogans.length];
    float hx = width * 0.5 + random(-40, 40);
    float hy = height * 0.5 + random(-40, 40);
    float hz = random(-200, 200);
    particles.add(new TextParticle(s, hx, hy, hz, i));
  }
}

// 预渲染文字到 PGraphics
void cacheText(String txt) {
  if (textCache.containsKey(txt)) return;
  
  PGraphics pg = createGraphics(200, 80, P2D);
  pg.beginDraw();
  pg.textFont(font);
  pg.textAlign(CENTER, CENTER);
  pg.textSize(24);
  pg.fill(0);
  pg.text(txt, 100, 40);
  pg.endDraw();
  
  textCache.put(txt, pg);
}

void draw() {
  background(245);
  
  noStroke();
  fill(245, 245, 245, 220);
  rect(0, 0, width, height);
  
  // 更新各个独立控制参数
  // 空格 = 靠近
  boolean spacePressed = keyPressed && key == ' ';
  targetProximity = spacePressed ? 1.0 : 0.0;
  depthProximity = lerp(depthProximity, targetProximity, 0.08);
  
  // X = 加速流动
  boolean xPressed = keyPressed && (key == 'x' || key == 'X');
  targetFlowSpeed = xPressed ? 3.0 : 1.0;
  flowSpeedMultiplier = lerp(flowSpeedMultiplier, targetFlowSpeed, 0.08);
  
  // Z = 增强抖动
  boolean zPressed = keyPressed && (key == 'z' || key == 'Z');
  targetJitter = zPressed ? 3.0 : 1.0;
  jitterMultiplier = lerp(jitterMultiplier, targetJitter, 0.08);
  
  // 靠近时呼吸更快更不安
  float breathSpeed = lerp(0.03, 0.08, depthProximity) * flowSpeedMultiplier;
  float breathIntensity = lerp(0.2, 0.5, depthProximity);
  globalPulse = 0.5 + 0.5 * sin(frameCount * breathSpeed) + breathIntensity * sin(frameCount * breathSpeed * 2.3);
  
  PVector hand = new PVector(mouseX, mouseY);
  float handActive = mousePressed ? 1.0 : 0.0;
  
  // 风场速度随靠近而加快
  float windSpeed = lerp(0.0012, 0.003, depthProximity) * flowSpeedMultiplier;
  noiseOffsetX += windSpeed;
  noiseOffsetY += windSpeed * 1.25;
  
  // 更新所有粒子
  for (TextParticle p : particles) {
    p.applyFlowField();
    p.applyHome(0.02);
    p.applyDisturb(hand, handActive);
    p.applyDepthProximity(depthProximity); // 新增：靠近效果
    p.update();
  }
  
  // 每 2 帧才排序一次
  if (frameCount % 2 == 0) {
    particles.sort((a, b) -> Float.compare(a.depth, b.depth));
  }
  
  // 绘制
  for (TextParticle p : particles) {
    p.display();
  }
  
  // FPS 显示 + 控制说明
  fill(0, 120);
  textSize(16);
  textAlign(LEFT, TOP);
  
  // 状态指示
  String proximityStatus = depthProximity > 0.5 ? "●" : "○";
  String flowStatus = flowSpeedMultiplier > 1.5 ? "●" : "○";
  String jitterStatus = jitterMultiplier > 1.5 ? "●" : "○";
  
  text("FPS: " + nf(frameRate, 1, 1) + " | Particles: " + MAX_PARTICLES + 
       "\n" + proximityStatus + " [SPACE] 靠近 → " + nf(depthProximity, 1, 2) +
       "\n" + flowStatus + " [X] 加速流动 → " + nf(flowSpeedMultiplier, 1, 2) + "x" +
       "\n" + jitterStatus + " [Z] 增强抖动 → " + nf(jitterMultiplier, 1, 2) + "x", 12, 10);
  textAlign(CENTER, CENTER);
}

class TextParticle {
  String content;
  PVector pos;
  PVector vel;
  PVector acc;
  PVector home;
  
  float depth;
  float baseSize;
  float baseAngle;
  int id;
  
  // 缓存计算结果
  float cachedAlpha;
  float cachedSize;
  float cachedAngle;
  
  TextParticle(String s, float hx, float hy, float hz, int _id) {
    content = s;
    home = new PVector(hx, hy, 0);
    pos = new PVector(hx + random(-10, 10), hy + random(-10, 10), 0);
    vel = new PVector();
    acc = new PVector();
    depth = hz;
    baseSize = random(14, 32);
    baseAngle = radians(random(-25, 25));
    id = _id;
  }
  
  void applyForce(PVector f) {
    acc.add(f);
  }
  
  void applyHome(float strength) {
    PVector dir = PVector.sub(home, pos);
    dir.mult(strength);
    applyForce(dir);
  }
  
  void applyFlowField() {
    float nx = (pos.x * 0.001) + noiseOffsetX + id * 0.01;
    float ny = (pos.y * 0.001) + noiseOffsetY + id * 0.01;
    float angle = noise(nx, ny) * TWO_PI * 2.0;
    
    PVector flow = new PVector(cos(angle), sin(angle));
    // 靠近时波动更强，受全局倍数影响
    float jitterAmt = lerp(0.3, 0.6, depthProximity) * jitterMultiplier;
    float speed = (0.5 + globalPulse * 0.8 + random(-jitterAmt, jitterAmt)) * flowSpeedMultiplier;
    flow.mult(speed);
    applyForce(flow);
    
    // 额外的随机抖动力 (靠近时更强，受全局倍数影响)
    float jitterForce = lerp(0.4, 0.8, depthProximity) * jitterMultiplier;
    PVector jitter = new PVector(random(-jitterForce, jitterForce), random(-jitterForce, jitterForce));
    applyForce(jitter);
  }
  
  // 新增：depth 靠近时的扩散效果
  void applyDepthProximity(float proximity) {
    if (proximity <= 0.0) return;
    
    // 从中心向外推
    PVector center = new PVector(width * 0.5, height * 0.5);
    PVector fromCenter = PVector.sub(pos, center);
    float dist = fromCenter.mag();
    
    if (dist > 1) {
      fromCenter.normalize();
      // 靠近时产生扩散力
      float spreadForce = proximity * 2.5;
      applyForce(PVector.mult(fromCenter, spreadForce));
    }
  }
  
  void applyDisturb(PVector hand, float active) {
    if (active <= 0.0) return;
    
    float radius = 350;
    
    float dx = pos.x - hand.x;
    float dy = pos.y - hand.y;
    float distSq = dx * dx + dy * dy;
    float radiusSq = radius * radius;
    
    if (distSq < radiusSq && distSq > 1) {
      float d = sqrt(distSq);
      float force = (radius - d) / radius;
      
      float invD = 1.0 / d;
      dx *= invD;
      dy *= invD;
      
      applyForce(new PVector(dx * force * 3.0, dy * force * 3.0));
      applyForce(new PVector(-dy * force * 1.8, dx * force * 1.8));
      
      depth = lerp(depth, 200, 0.1);
    }
  }
  
  void update() {
    vel.add(acc);
    vel.mult(0.90);
    pos.add(vel);
    acc.mult(0);
    
    // 边界检查
    if (pos.x < -200 || pos.x > width + 200 || pos.y < -200 || pos.y > height + 200) {
      pos.x = width * 0.5 + random(-60, 60);
      pos.y = height * 0.5 + random(-60, 60);
      vel.mult(0);
    }
    
    depth = lerp(depth, 0, 0.01);
    
    // 预计算显示参数
    float depthNorm = constrain(map(depth, -200, 200, 0.6, 1.4), 0.6, 1.5);
    cachedAlpha = map(depthNorm, 0.6, 1.5, 80, 220);
    cachedSize = baseSize * depthNorm;
    
    // 靠近时抖动更剧烈，受全局倍数影响
    float jitterSpeed = lerp(0.12, 0.25, depthProximity) * flowSpeedMultiplier;
    float jitterAmp = lerp(8.0, 15.0, depthProximity) * jitterMultiplier;
    float jitterAngle = radians(
      sin((frameCount + id * 13) * jitterSpeed) * jitterAmp + 
      cos((frameCount + id * 7) * jitterSpeed * 1.5) * jitterAmp * 0.6
    );
    cachedAngle = baseAngle + jitterAngle;
  }
  
  void display() {
    pushMatrix();
    translate(pos.x, pos.y);
    rotate(cachedAngle);
    
    // 使用缓存的文字图像
    tint(0, cachedAlpha);
    float scale = cachedSize / 24.0;
    scale(scale);
    
    PGraphics cached = textCache.get(content);
    if (cached != null) {
      imageMode(CENTER);
      image(cached, 0, 0);
    }
    
    noTint();
    popMatrix();
  }
}
