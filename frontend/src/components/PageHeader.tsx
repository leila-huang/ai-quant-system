import { Space, Typography, Button, Divider, Breadcrumb } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const { Title, Text } = Typography;

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  extra?: React.ReactNode;
  onBack?: () => void;
  backable?: boolean;
  breadcrumbs?: Array<{
    title: React.ReactNode;
    href?: string;
  }>;
  className?: string;
}

/**
 * 页面头部组件
 * 提供统一的页面标题、操作按钮、面包屑等功能
 */
const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  subtitle,
  extra,
  onBack,
  backable = false,
  breadcrumbs,
  className,
}) => {
  const navigate = useNavigate();

  const handleBack = () => {
    if (onBack) {
      onBack();
    } else {
      navigate(-1);
    }
  };

  return (
    <div className={className}>
      {/* 面包屑导航 */}
      {breadcrumbs && breadcrumbs.length > 0 && (
        <div style={{ marginBottom: '16px' }}>
          <Breadcrumb
            items={breadcrumbs.map(item => ({
              title: item.href ? (
                <a
                  href={item.href}
                  onClick={e => {
                    e.preventDefault();
                    navigate(item.href!);
                  }}
                >
                  {item.title}
                </a>
              ) : (
                item.title
              ),
            }))}
          />
        </div>
      )}

      {/* 页面标题区域 */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: subtitle ? '8px' : '24px',
        }}
      >
        <Space size="middle">
          {(backable || onBack) && (
            <Button
              type="text"
              icon={<ArrowLeftOutlined />}
              onClick={handleBack}
              style={{
                fontSize: '16px',
                height: '32px',
                width: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            />
          )}

          <Title level={2} style={{ margin: 0 }}>
            {title}
          </Title>
        </Space>

        {/* 额外操作区域 */}
        {extra && <Space>{extra}</Space>}
      </div>

      {/* 副标题 */}
      {subtitle && (
        <div style={{ marginBottom: '24px' }}>
          <Text type="secondary" style={{ fontSize: '14px' }}>
            {subtitle}
          </Text>
        </div>
      )}

      <Divider style={{ margin: '0 0 24px 0' }} />
    </div>
  );
};

export default PageHeader;

