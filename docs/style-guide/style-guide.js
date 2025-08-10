// Style Guide Interactive Examples
sap.ui.getCore().attachInit(function() {
    // Button Examples
    const buttonContainer = new sap.m.HBox({
        items: [
            new sap.m.Button({ text: "Default" }),
            new sap.m.Button({ text: "Emphasized", type: sap.m.ButtonType.Emphasized }),
            new sap.m.Button({ text: "Accept", type: sap.m.ButtonType.Accept }),
            new sap.m.Button({ text: "Reject", type: sap.m.ButtonType.Reject }),
            new sap.m.Button({ text: "Ghost", type: sap.m.ButtonType.Ghost }),
            new sap.m.Button({ text: "Transparent", type: sap.m.ButtonType.Transparent })
        ],
        wrap: sap.m.FlexWrap.Wrap
    });
    buttonContainer.placeAt("buttonExamples");

    // Card Example
    const card = new sap.f.Card({
        width: "300px",
        header: new sap.f.cards.Header({
            title: "Agent Performance",
            subtitle: "Last 24 hours",
            iconSrc: "sap-icon://line-chart"
        }),
        content: new sap.f.cards.NumericContent({
            value: "98.5",
            valueColor: sap.m.ValueColor.Good,
            scale: "%",
            indicator: sap.m.DeviationIndicator.Up,
            sideIndicators: [
                new sap.f.cards.NumericSideIndicator({
                    title: "Target",
                    number: "95",
                    unit: "%"
                })
            ]
        })
    });
    card.placeAt("cardExample");

    // Form Example
    const form = new sap.ui.layout.form.SimpleForm({
        editable: true,
        layout: "ResponsiveGridLayout",
        content: [
            new sap.m.Label({ text: "Agent Name" }),
            new sap.m.Input({ value: "Processing Agent 01", width: "100%" }),
            new sap.m.Label({ text: "Type" }),
            new sap.m.Select({
                width: "100%",
                items: [
                    new sap.ui.core.Item({ key: "1", text: "Processing" }),
                    new sap.ui.core.Item({ key: "2", text: "Analytics" }),
                    new sap.ui.core.Item({ key: "3", text: "Integration" })
                ]
            }),
            new sap.m.Label({ text: "Status" }),
            new sap.m.Switch({ state: true }),
            new sap.m.Label({ text: "Description" }),
            new sap.m.TextArea({ 
                value: "High-performance data processing agent",
                width: "100%",
                rows: 3
            })
        ]
    });
    form.placeAt("formExample");

    // Animation Examples - Add click handlers
    document.querySelectorAll('.animation-box').forEach(box => {
        box.addEventListener('click', function() {
            const animationClass = this.classList[1];
            this.style.animation = 'none';
            setTimeout(() => {
                this.style.animation = '';
            }, 10);
        });
    });

    // Smooth scrolling for navigation
    document.querySelectorAll('.style-guide-nav a').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // Highlight active section on scroll
    const sections = document.querySelectorAll('.style-section');
    const navLinks = document.querySelectorAll('.style-guide-nav a');

    function highlightNavigation() {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.style.borderBottomColor = 'transparent';
            if (link.getAttribute('href').substring(1) === current) {
                link.style.borderBottomColor = 'var(--sapBrandColor)';
            }
        });
    }

    window.addEventListener('scroll', highlightNavigation);
    highlightNavigation();

    // Copy color values on click
    document.querySelectorAll('.color-swatch').forEach(swatch => {
        swatch.style.cursor = 'pointer';
        swatch.addEventListener('click', function() {
            const colorVar = this.querySelector('.color-var').textContent;
            navigator.clipboard.writeText(colorVar).then(() => {
                sap.m.MessageToast.show(`Copied: ${colorVar}`);
            });
        });
    });

    // Interactive pattern examples
    const patternExamples = {
        'master-detail': `
            <SplitApp id="splitApp">
                <masterPages>
                    <Page title="Items">
                        <List items="{/items}">
                            <StandardListItem 
                                title="{name}"
                                description="{description}"
                                type="Navigation"
                                press="onItemPress"/>
                        </List>
                    </Page>
                </masterPages>
                <detailPages>
                    <Page title="{selectedItem>/name}">
                        <ObjectHeader
                            title="{selectedItem>/name}"
                            number="{selectedItem>/id}"
                            numberUnit="ID">
                            <attributes>
                                <ObjectAttribute 
                                    title="Status" 
                                    text="{selectedItem>/status}"/>
                            </attributes>
                        </ObjectHeader>
                    </Page>
                </detailPages>
            </SplitApp>
        `,
        'worklist': `
            <Table
                items="{/workItems}"
                mode="MultiSelect">
                <headerToolbar>
                    <Toolbar>
                        <Title text="Work Items ({/workItems/length})"/>
                        <ToolbarSpacer/>
                        <Button text="Process Selected" press="onProcessItems"/>
                    </Toolbar>
                </headerToolbar>
                <columns>
                    <Column><Text text="ID"/></Column>
                    <Column><Text text="Name"/></Column>
                    <Column><Text text="Status"/></Column>
                    <Column><Text text="Priority"/></Column>
                </columns>
                <items>
                    <ColumnListItem>
                        <cells>
                            <Text text="{id}"/>
                            <Text text="{name}"/>
                            <ObjectStatus text="{status}" state="{statusState}"/>
                            <ObjectNumber number="{priority}" state="{priorityState}"/>
                        </cells>
                    </ColumnListItem>
                </items>
            </Table>
        `
    };

    // Generate pattern example images or placeholders
    const patternSections = document.querySelectorAll('.pattern-example');
    patternSections.forEach(section => {
        if (!section.querySelector('img').src) {
            // Create placeholder if image doesn't exist
            const placeholder = document.createElement('div');
            placeholder.style.cssText = `
                width: 100%;
                height: 300px;
                background: var(--sapGroup_ContentBackground);
                border: 1px solid var(--sapGroup_ContentBorderColor);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: var(--sapFontLargeSize);
                color: var(--sapContent_LabelColor);
                margin-bottom: 16px;
            `;
            placeholder.textContent = 'Pattern Example Preview';
            section.querySelector('img').replaceWith(placeholder);
        }
    });
});