#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Basic translations for all languages
const translations = {
  en: {
    welcome: "Welcome to A2A Network",
    agents: "Agents",
    services: "Services",
    marketplace: "Marketplace",
    dashboard: "Dashboard",
    settings: "Settings",
    blockchain: "Blockchain",
    contracts: "Smart Contracts",
    operations: "Operations",
    notifications: "Notifications",
    login: "Login",
    logout: "Logout",
    save: "Save",
    cancel: "Cancel",
    search: "Search",
    loading: "Loading..."
  },
  de: {
    welcome: "Willkommen bei A2A Network",
    agents: "Agenten",
    services: "Dienste",
    marketplace: "Marktplatz",
    dashboard: "Dashboard",
    settings: "Einstellungen",
    blockchain: "Blockchain",
    contracts: "Smart Contracts",
    operations: "Betrieb",
    notifications: "Benachrichtigungen",
    login: "Anmelden",
    logout: "Abmelden",
    save: "Speichern",
    cancel: "Abbrechen",
    search: "Suchen",
    loading: "Laden..."
  },
  fr: {
    welcome: "Bienvenue sur A2A Network",
    agents: "Agents",
    services: "Services",
    marketplace: "Marché",
    dashboard: "Tableau de bord",
    settings: "Paramètres",
    blockchain: "Blockchain",
    contracts: "Contrats intelligents",
    operations: "Opérations",
    notifications: "Notifications",
    login: "Connexion",
    logout: "Déconnexion",
    save: "Enregistrer",
    cancel: "Annuler",
    search: "Rechercher",
    loading: "Chargement..."
  },
  es: {
    welcome: "Bienvenido a A2A Network",
    agents: "Agentes",
    services: "Servicios",
    marketplace: "Mercado",
    dashboard: "Panel de control",
    settings: "Configuración",
    blockchain: "Blockchain",
    contracts: "Contratos inteligentes",
    operations: "Operaciones",
    notifications: "Notificaciones",
    login: "Iniciar sesión",
    logout: "Cerrar sesión",
    save: "Guardar",
    cancel: "Cancelar",
    search: "Buscar",
    loading: "Cargando..."
  },
  ja: {
    welcome: "A2Aネットワークへようこそ",
    agents: "エージェント",
    services: "サービス",
    marketplace: "マーケットプレイス",
    dashboard: "ダッシュボード",
    settings: "設定",
    blockchain: "ブロックチェーン",
    contracts: "スマートコントラクト",
    operations: "オペレーション",
    notifications: "通知",
    login: "ログイン",
    logout: "ログアウト",
    save: "保存",
    cancel: "キャンセル",
    search: "検索",
    loading: "読み込み中..."
  },
  zh: {
    welcome: "欢迎来到A2A网络",
    agents: "代理",
    services: "服务",
    marketplace: "市场",
    dashboard: "仪表板",
    settings: "设置",
    blockchain: "区块链",
    contracts: "智能合约",
    operations: "操作",
    notifications: "通知",
    login: "登录",
    logout: "登出",
    save: "保存",
    cancel: "取消",
    search: "搜索",
    loading: "加载中..."
  }
};

// For other languages, use English as fallback
const fallbackLanguages = ['it', 'pt', 'ru', 'ar', 'he', 'hi', 'nl', 'pl', 'tr', 'cs', 'sv', 'da', 'no', 'fi', 'ko', 'zh-TW'];

const localesDir = path.join(__dirname, '../srv/i18n/locales');

// Update main language files
Object.entries(translations).forEach(async ([lang, content]) => {
  const filePath = path.join(localesDir, `${lang}.json`);
  await fs.writeFile(filePath, JSON.stringify(content), 'utf8');
  console.log(`Updated ${lang}.json`);
});

// Update fallback language files with English content
fallbackLanguages.forEach(async (lang) => {
  const filePath = path.join(localesDir, `${lang}.json`);
  await fs.writeFile(filePath, JSON.stringify(translations.en), 'utf8');
  console.log(`Updated ${lang}.json with English fallback`);
});

console.log('All translation files updated successfully!');