import oscP5.*;
import netP5.*;

// --- OSC 通信模块 ---
OscP5 oscP5;

// --- 核心情绪变量 (来自 Wekinator) ---
// 1. 窒息感 (0.0 - 1.0): 控制雾霾浓度、扩散、变白
float Smog_Density = 0.0;
// 2. 恐慌感 (0.0 - 1.0): 控制流速、抖动、混乱
float System_Entropy = 0.0;

// 手的位置 (来自 Wekinator 或 Python 直通)
PVector hand = new PVector(0, 0);
float handActive = 0.0; // 手是否激活 (有数据时为1)

// --- 粒子系统变量 ---
ArrayList<TextParticle> particles;
PFont font;
int MAX_PARTICLES;
float globalPulse = 0;

float noiseOffsetX = 0.0;
float noiseOffsetY = 100.0;

// 文字缓存
HashMap<String, PGraphics> textCache;

// 视觉参数 (将由情绪变量平滑驱动)
float depthProximity = 0.0; 
float targetProximity = 0.0;

float flowSpeedMultiplier = 1.0; 
float targetFlowSpeed = 1.0;

float jitterMultiplier = 1.0; 
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
  
  // 初始化 OSC，监听端口 12000 (Wekinator 发送端口)
  oscP5 = new OscP5(this, 12000);
  
  // 字体设置
  String[] fontOptions = {
    "Microsoft YaHei", "SimHei", "PingFang SC", 
    "Noto Sans CJK SC", "Arial Unicode MS", "Sans Serif"
  };
  
  font = null;
  for (String fontName : fontOptions) {
    try {
      font = createFont(fontName, 24, true);
      if (font != null) {
        println("Using font: " + fontName);
        break;
      }
    } catch (Exception e) { continue; }
  }
  if (font == null) font = createFont("Sans Serif", 24, true);
  textFont(font);
  
  particles = new ArrayList<TextParticle>();
  textCache = new HashMap<String, PGraphics>();
  
  // 初始化粒子
  float density = 0.00016;
  MAX_PARTICLES = int(width * height * density);
  MAX_PARTICLES = min(MAX_PARTICLES, 380);
  
  for (String s : slogans) {
    cacheText(s);
  }
  
  for (int i = 0; i < MAX_PARTICLES; i++) {
    String s = slogans[i % slogans.length];
    float hx = width * 0.5 + random(-40, 40);
    float hy = height * 0.5 + random(-40, 40);
    float hz = random(-200, 200);
    particles.add(new TextParticle(s, hx, hy, hz, i));
  }
  
  // 初始化手的位置在中心
  hand.set(width/2, height/2);
}

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
  
  // --- 1. 核心映射逻辑：将 AI 情绪变量映射到视觉参数 ---
  
  // A. 窒息感 (Smog_Density) -> 影响 靠近扩散程度
  targetProximity = Smog_Density; 
  depthProximity = lerp(depthProximity, targetProximity, 0.05);
  
  // B. 恐慌感 (System_Entropy) -> 影响 流速 + 抖动
  // 恐慌越高，流速越快 (1.0 -> 4.0 倍)
  targetFlowSpeed = map(System_Entropy, 0, 1, 1.0, 4.0);
  flowSpeedMultiplier = lerp(flowSpeedMultiplier, targetFlowSpeed, 0.05);
  
  // 恐慌越高，抖动越剧烈 (1.0 -> 6.0 倍)
  targetJitter = map(System_Entropy, 0, 1, 1.0, 6.0);
  jitterMultiplier = lerp(jitterMultiplier, targetJitter, 0.05);
  
  // 呼吸节奏随靠近变快
  float breathSpeed = lerp(0.03, 0.1, depthProximity) * flowSpeedMultiplier;
  float breathIntensity = lerp(0.2, 0.5, depthProximity);
  globalPulse = 0.5 + 0.5 * sin(frameCount * breathSpeed) + breathIntensity * sin(frameCount * breathSpeed * 2.3);
  
  // 风场更新
  float windSpeed = lerp(0.0012, 0.005, depthProximity) * flowSpeedMultiplier;
  noiseOffsetX += windSpeed;
  noiseOffsetY += windSpeed * 1.25;
  
  // 更新粒子
  for (TextParticle p : particles) {
    p.applyFlowField();
    
    // 如果没有手势交互 (handActive < 0.1)，且方差低(Entropy低)，则稍微聚拢中心
    // 这是“信息茧房”效果
    float homeForce = map(System_Entropy, 0, 1, 0.02, -0.01); 
    if (homeForce > 0) p.applyHome(homeForce);
    
    // 手势推开 (Interaction)
    p.applyDisturb(hand, 1.0); // 始终开启推开效果，只要 hand 坐标在变
    
    p.applyDepthProximity(depthProximity); 
    p.update();
  }
  
  if (frameCount % 2 == 0) {
    particles.sort((a, b) -> Float.compare(a.depth, b.depth));
  }
  
  for (TextParticle p : particles) {
    p.display();
  }
  
  // --- Debug / 状态显示 ---
  fill(0, 180);
  textSize(16);
  textAlign(LEFT, TOP);
  text("INFO SMOG SYSTEM | AI DRIVEN", 20, 20);
  
  textSize(14);
  text("Density (Suffocation): " + nf(Smog_Density, 1, 2), 20, 50);
  text("Entropy (Panic): " + nf(System_Entropy, 1, 2), 20, 70);
  
  // 可视化手的位置
  noFill();
  stroke(255, 0, 0, 150);
  ellipse(hand.x, hand.y, 20, 20);
}

// --- OSC 事件接收 ---
void oscEvent(OscMessage theOscMessage) {
  // 检查是否是 Wekinator 发来的消息
  if (theOscMessage.checkAddrPattern("/wek/outputs")) {
    
    // Output 1: 窒息感 (0.0 - 1.0)
    if(theOscMessage.checkTypetag("ffff")) { // 确保有4个浮点数
       Smog_Density = theOscMessage.get(0).floatValue();
       
       // Output 2: 恐慌感 (0.0 - 1.0)
       System_Entropy = theOscMessage.get(1).floatValue();
       
       // Output 3: 手势 X (0.0 - 1.0)
       float raw_x = theOscMessage.get(2).floatValue();
       
       // Output 4: 手势 Y (0.0 - 1.0)
       float raw_y = theOscMessage.get(3).floatValue();
       
       // 映射坐标 (注意 X 轴镜像翻转，符合镜面交互直觉)
       hand.x = (1.0 - raw_x) * width;
       hand.y = raw_y * height;
       
       handActive = 1.0; 
    }
  }
}

class TextParticle {
  String content;
  PVector pos, vel, acc, home;
  float depth, baseSize, baseAngle;
  int id;
  float cachedAlpha, cachedSize, cachedAngle;
  
  TextParticle(String s, float hx, float hy, float hz, int _id) {
    content = s;
    home = new PVector(hx, hy, 0);
    pos = new PVector(hx + random(-10, 10), hy + random(-10, 10), 0);
    vel = new PVector(); acc = new PVector();
    depth = hz;
    baseSize = random(14, 32);
    baseAngle = radians(random(-25, 25));
    id = _id;
  }
  
  void applyForce(PVector f) { acc.add(f); }
  
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
    float jitterAmt = lerp(0.3, 0.6, depthProximity) * jitterMultiplier;
    float speed = (0.5 + globalPulse * 0.8 + random(-jitterAmt, jitterAmt)) * flowSpeedMultiplier;
    flow.mult(speed);
    applyForce(flow);
    
    float jitterForce = lerp(0.4, 0.8, depthProximity) * jitterMultiplier;
    PVector jitter = new PVector(random(-jitterForce, jitterForce), random(-jitterForce, jitterForce));
    applyForce(jitter);
  }
  
  void applyDepthProximity(float proximity) {
    if (proximity <= 0.01) return;
    PVector center = new PVector(width * 0.5, height * 0.5);
    PVector fromCenter = PVector.sub(pos, center);
    float dist = fromCenter.mag();
    if (dist > 1) {
      fromCenter.normalize();
      float spreadForce = proximity * 2.5;
      applyForce(PVector.mult(fromCenter, spreadForce));
    }
  }
  
  void applyDisturb(PVector handPos, float active) {
    if (active <= 0.0) return;
    float radius = 350;
    float dx = pos.x - handPos.x;
    float dy = pos.y - handPos.y;
    float distSq = dx * dx + dy * dy;
    
    if (distSq < radius * radius && distSq > 1) {
      float d = sqrt(distSq);
      float force = (radius - d) / radius;
      float invD = 1.0 / d;
      dx *= invD; dy *= invD;
      applyForce(new PVector(dx * force * 3.0, dy * force * 3.0));
      depth = lerp(depth, 200, 0.1);
    }
  }
  
  void update() {
    vel.add(acc);
    vel.mult(0.90);
    pos.add(vel);
    acc.mult(0);
    
    if (pos.x < -200 || pos.x > width + 200 || pos.y < -200 || pos.y > height + 200) {
      pos.x = width * 0.5 + random(-60, 60);
      pos.y = height * 0.5 + random(-60, 60);
      vel.mult(0);
    }
    
    depth = lerp(depth, 0, 0.01);
    float depthNorm = constrain(map(depth, -200, 200, 0.6, 1.4), 0.6, 1.5);
    
    // --- 视觉反馈核心：透明度随窒息感 (Proximity) 变化 ---
    // 基础透明度 + 靠近时的额外增亮
    float baseAlpha = map(depthNorm, 0.6, 1.5, 80, 220);
    float proximityAlpha = lerp(1.0, 1.8, depthProximity); // 靠近时更白
    cachedAlpha = constrain(baseAlpha * proximityAlpha, 0, 255);
    
    cachedSize = baseSize * depthNorm;
    
    float jitterSpeed = lerp(0.12, 0.25, depthProximity) * flowSpeedMultiplier;
    float jitterAmp = lerp(8.0, 15.0, depthProximity) * jitterMultiplier;
    float jitterAngle = radians(sin((frameCount + id * 13) * jitterSpeed) * jitterAmp);
    cachedAngle = baseAngle + jitterAngle;
  }
  
  void display() {
    pushMatrix();
    translate(pos.x, pos.y);
    rotate(cachedAngle);
    tint(0, cachedAlpha); // 使用计算好的透明度
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
